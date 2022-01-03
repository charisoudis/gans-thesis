import gc
import math
import sys
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor

from modules.discriminators.stylegan import StyleGanDiscriminator
from modules.partial.encoding import NoiseMappingNetwork
from modules.partial.normalization import AdaptiveInstanceNorm2d
from utils.command_line_logger import CommandLineLogger
from utils.distributions import TruncatedStandardNormal
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import get_total_params, enable_verbose
from utils.string import to_human_readable
from utils.train import get_alpha_curve


class InjectNoise(nn.Module):
    """
    InjectNoise Class:
    This class implements the random noise injection that occurs before every AdaIN block of the original StyleGAN or
    before every ModulatedConv2d layer of StyleGANv2.
    """

    def __init__(self, c_in: int):
        """
        InjectNoise class constructor.
        :param c_in: the number of channels of the expected input tensor
        """
        super().__init__()
        # Initiate the weights for the channels from a random normal distribution
        # You use nn.Parameter so that these weights can be optimized
        self.weight = nn.Parameter(torch.randn(c_in).view(1, c_in, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of InjectNoise: Given an image, returns the image plus random noise.
        :param x: the input feature map of shape (N, C_in, H, W)
        :return: a torch.Tensor object of the same shape as $x$
        """
        # Noise :   (N,1,W,H)
        # Weight:   (N,C,1,1)
        batch_size, c_in, h, w = x.shape
        noise_shape = (batch_size, 1, w, h)
        noise = torch.randn(noise_shape, device=x.device)
        return x + self.weight * noise


class StyleGanGeneratorBlock(nn.Module):
    """
    StyleGanGeneratorBlock Class.
    This class implements a single block of the StyleGAN v1 generator.
    """

    def __init__(self, c_in: int, c_out: int, w_dim: int = 512, kernel_size: int = 3,
                 kernel_size_2: Optional[int] = None, padding=1, padding_2: Optional = None):
        """
        StyleGanGeneratorBlock class constructor.
        :param (int) c_in: number of channels in the input
        :param (int) c_out: number of channels expected in the output
        :param (int) w_dim: size of the intermediate noise vector (defaults to 512)
        :param (int) kernel_size: nn.Conv2d()'s kernel_size argument (defaults to 3)
        :param (int) kernel_size_2: 2nd nn.Conv2d()'s kernel_size argument or None to use the same as the 1st Conv2d's
        """
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        # noinspection PyTypeChecker
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding),
            nn.LeakyReLU(0.2),
            InjectNoise(c_out),
        )
        self.adaptive_norm1 = AdaptiveInstanceNorm2d(c_out, w_dim)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size_2 if kernel_size_2 is not None else kernel_size,
                      padding=padding_2 if padding_2 is not None else padding),
            nn.LeakyReLU(0.2),
            InjectNoise(c_out),
        )
        self.adaptive_norm2 = AdaptiveInstanceNorm2d(c_out, w_dim)
        # self.pixel_norm = PixelNorm2d()

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None):
        """
        Function for completing a forward pass of StyleGanGeneratorBlock:
        Given an input x and the style vector w, it computes a StyleGAN generator block.
        :param (torch.Tensor) x: the input to the generator's block, of shape (N,C_in,H,W)
        :param (torch.Tensor) w: the intermediate noise vector, of shape (N, W_dim)
        :return a torch.Tensor object of shape (N, C_out, H, W)
        """
        x = self.conv_block1(x)
        if w is not None:
            x = self.adaptive_norm1(x, w)
        else:
            x = self.pixel_norm(x)
        x = self.conv_block2(x)
        if w is not None:
            return self.adaptive_norm2(x, w)
        return self.pixel_norm(x)


class StyleGanGenerator(nn.Module, BalancedFreezable, Verbosable):
    """
    StyleGanGenerator Class.
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        c_in: the dimension of the constant input, usually w_dim, a scalar
        c_out: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        c_hidden: the inner dimension, a scalar
    """

    def __init__(self, z_dim: int = 512, map_hidden_dim: int = 512, map_n_blocks: int = 4, w_dim: int = 512,
                 c_const: int = 512, c_hidden: int = 512, c_out: int = 3, resolution: int = 128,
                 kernel_size: int = 3, alpha_multiplier: float = 10.0, num_iters: int = 1,
                 logger: Optional[CommandLineLogger] = None, truncation: Optional[float] = None):
        """
        StyleGanGenerator class constructor.
        :param (int) z_dim: size of the noise vector (defaults to 512)
        :param (int) map_hidden_dim: hidden size of NoiseMappingNetwork's FC layers (defaults to 512)
        :param (int) map_n_blocks: number of blocks (FC+ReLU) in NoiseMappingNetwork (defaults to 3)
        :param (int) w_dim: size of the intermediate noise vector (defaults to 512)
        :param (int) c_const: number of channels of the 4x4 constant input at the beginning of the generator
                              (defaults to 512)
        :param (int) c_out: number of channels expected in the output
        :param (int) resolution: width and height of the desired output
        :param (int) kernel_size: nn.Conv2d()'s kernel_size argument (defaults to 3)
        :param (float) alpha_multiplier: multiplier that defines the smoothness of the alpha curve
                                         (1=linear, ..., 10=smooth, ..., 1000=delta)
        :param (int) num_iters: number of iterations
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        :param (optional) truncation: a float from 0 to 1 to truncate z-space normal prior
        """
        # Save arguments
        self.locals = locals()
        del self.locals['self']
        self.z_dim = z_dim

        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Init nn.Module
        nn.Module.__init__(self)
        # Noise mapping network
        self.noise_mapping = NoiseMappingNetwork(z_dim=z_dim, hidden_dim=map_hidden_dim, n_blocks=map_n_blocks,
                                                 w_dim=w_dim)
        # Starting Constant: typically this constant is initiated to all ones
        self.constant = nn.Parameter(torch.ones(1, c_const, 4, 4))
        # Generator convolutional blocks
        self.block0 = StyleGanGeneratorBlock(c_in=c_const, c_out=c_hidden, w_dim=w_dim, kernel_size=4,
                                             kernel_size_2=kernel_size, padding='same', padding_2=1)
        # noinspection PyTypeChecker
        self.block0_toRGB = nn.Conv2d(c_hidden, c_out, kernel_size=1)
        for bi in range(3, int(math.log2(resolution)) + 1):
            block_index = bi - 2
            _c_hidden_dict = {
                4: c_hidden,
                8: c_hidden,
                16: c_hidden,
                32: c_hidden,
                64: c_hidden // 2,
                128: c_hidden // 4,
                256: c_hidden // 8,
                512: c_hidden // 16,
            }
            _c_hidden_prev = _c_hidden_dict[2 ** (bi - 1)]
            _c_hidden = _c_hidden_dict[2 ** bi]

            # print('bi', bi, '_c_hidden', _c_hidden)
            setattr(self, f'upsample{block_index}', nn.Upsample(size=2 ** bi, mode='bilinear', align_corners=False))
            setattr(self, f'block{block_index}',
                    StyleGanGeneratorBlock(c_in=_c_hidden_prev, c_out=_c_hidden, w_dim=w_dim, kernel_size=kernel_size))
            if bi == int(math.log2(resolution)):
                # noinspection PyTypeChecker
                setattr(self, f'upsample{block_index}_toRGB', nn.Conv2d(_c_hidden_prev, c_out, kernel_size=1))
                # noinspection PyTypeChecker
                setattr(self, f'block{block_index}_toRGB', nn.Conv2d(_c_hidden, c_out, kernel_size=1))
        # Save args
        self.resolution = self.locals['resolution']

        # Fade coefficient
        self.alpha_curve = get_alpha_curve(num_iters=num_iters, alpha_multiplier=alpha_multiplier)
        self.alpha_index = 0
        self.alpha = self.alpha_curve[self.alpha_index]

        # Truncation trick
        if truncation is not None:
            self.dist = TruncatedStandardNormal(a=-truncation, b=truncation)
        else:
            self.dist = None

        self.logger = logger if logger is not None else \
            CommandLineLogger(log_level='debug', name=self.__class__.__name__)

    def forward(self, z: Optional[torch.Tensor] = None, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        :param (torch.tensor) z: input noise vector of shape (N, z_dim)
        :param (optional) w: directly pass the W-vector
        :return: a torch.Tensor object containing the mixing of the output of the UpSampler and the last conv block
        """
        # Get current alpha value
        alpha = self.alpha
        epsilon = sys.float_info.epsilon
        ################################################################################################################
        ################################################# DEV LOGGING ##################################################
        ################################################################################################################
        # if abs(alpha - 0.5) < 1e-3:
        #     self.logger.debug(f'[GEN] alpha={alpha} (alpha_index={self.alpha_index}, len()={len(self.alpha_curve)})')
        # elif abs(alpha - 0.99) < 1e-3:
        #     self.logger.debug(f'[GEN] alpha={alpha} (alpha_index={self.alpha_index}, len()={len(self.alpha_curve)})')
        #
        # return torch.zeros(z.shape[0], 3, self.resolution, self.resolution, device=z.device, requires_grad=True)
        ################################################################################################################

        # Get total number of blocks
        resolution_log2 = int(math.log2(self.resolution))
        # Map noise
        if w is None:
            assert z is not None, 'z and w cannot be None simultaneously'
            w = self.noise_mapping(z)  # (N, w_dim)
        # Pass through generator blocks
        #   - constant + first block
        x = self.constant.repeat((w.shape[0], 1, 1, 1))
        x = self.block0(x, w)
        ################################################################################################################
        ################################################# DEV LOGGING ##################################################
        ################################################################################################################
        # self.logger.debug(f'x: {x.shape} <-- block0(c_in={self.block0.c_in},c_o={self.block0.c_out}, w) '
        #                   f'<-- constant: {self.constant.shape}')
        ################################################################################################################
        if resolution_log2 == 2:
            return self.block0_toRGB(x)
        #   - upsampling + conv blocks until the last (wherein the mixing happens)
        for bi in range(3, resolution_log2 + 1):
            block_index = bi - 2
            upsample = getattr(self, f'upsample{block_index}')
            # x_shape = x.shape
            x = upsample(x)
            ############################################################################################################
            ############################################### DEV LOGGING ################################################
            ############################################################################################################
            # self.logger.debug(f'x: {x.shape} <-- upsample{block_index} <-- x: {x_shape}')
            ############################################################################################################
            block = getattr(self, f'block{block_index}')
            if bi == int(math.log2(self.resolution)):
                upsample_toRGB = getattr(self, f'upsample{block_index}_toRGB')
                block_toRGB = getattr(self, f'block{block_index}_toRGB')
                ########################################################################################################
                ############################################# DEV LOGGING ##############################################
                ########################################################################################################
                # self.logger.debug(f'x <- alpha={alpha} - block{block_index}_toRGB(c_h={block_toRGB.in_channels},'
                #                   f'c_o={block_toRGB.out_channels}) <-- block{block_index}(c_in={block.c_in},'
                #                   f'c_o={block.c_out}, w) <-- x: {x.shape}')
                ########################################################################################################

                ########################################################################################################
                ############################################# DEV LOGGING ##############################################
                ########################################################################################################
                # self.logger.debug(f'x_upsampled <- 1-alpha={(1 - alpha)} - upsample{block_index}_toRGB('
                #                   f'c_in={upsample_toRGB.in_channels},c_o={upsample_toRGB.out_channels}) <-- '
                #                   f'x: {x.shape}')
                ########################################################################################################
                if alpha + epsilon < 1.0:
                    x_upsampled = upsample_toRGB(x.clone())
                    x = block(x, w)
                    x = block_toRGB(x)
                    return (1.0 - alpha) * x_upsampled + alpha * x
                x = block(x, w)
                return block_toRGB(x)
            # xs = x.shape
            x = block(x, w)
            ############################################################################################################
            ############################################### DEV LOGGING ################################################
            ############################################################################################################
            # self.logger.debug(f'x: {x.shape} <-- block{block_index}(c_in={block.c_in},c_o={block.c_out}) <-- x: {xs}')
            ############################################################################################################
        return x

    def get_noise(self, batch_size: int, device: Optional[str or torch.device] = None) -> torch.Tensor:
        """
        Create and return a new noise vector in the same device as the model itself.
        :param (int) batch_size: number of vectors in batch
        :param (str or torch.device or None) device: vector's device
        :return: a torch.Tensor object of shape (N, z_dim)
        """
        if self.dist is None:
            return torch.randn(batch_size, self.locals['z_dim'], device=device)
        return self.dist.sample((batch_size, self.locals['z_dim'])).to(device)

    def get_loss(self, batch_size: int, disc: StyleGanDiscriminator,
                 adv_criterion: Optional[nn.modules.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the loss of the generator given inputs. If the criteria are not provided they will be set using the
        instance's given (or default) configuration.
        :param (int) batch_size: current number of input noise vectors
        :param (torch.nn.Module) disc: the StyleGAN Discriminator network
        :param (optional) adv_criterion: the adversarial loss function; takes the discriminator predictions and the
                                         target labels and returns a adversarial loss (which we aim to minimize)
        :return: a torch.Tensor object containing the Generator's loss (a scalar) and the output batch of images (for
                 visualization purposes)
        """
        # 0) Create a batch of noise vectors
        z = self.get_noise(batch_size=batch_size, device=self.constant.device)
        # 1) Make a forward pass on the Generator
        fake = self(z)
        # 2) Compute Generator's Adversarial Loss
        gen_loss = disc.get_loss_on_batch(fake, is_real=True, criterion=adv_criterion)
        return gen_loss, fake.detach()

    def get_layer_attr_names(self) -> List[str]:
        layer_names = ['block0', 'block0_toRGB']
        for bi in range(3, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            layer_names.append(f'upsample{block_index}')
            layer_names.append(f'block{block_index}')
            if bi == int(math.log2(self.resolution)):
                layer_names.append(f'upsample{block_index}_toRGB')
                layer_names.append(f'block{block_index}_toRGB')
        return layer_names

    def plot_alpha_curve(self):
        y = self.alpha_curve
        x = np.arange(len(y))
        plt.plot(x, y, '-.')
        plt.title(f'GEN @ {self.resolution}x{self.resolution} (num_iters={self.locals["num_iters"]})')
        plt.show()

    def grow(self, num_iters: int = 2, device: Optional[str or torch.device] = None) -> 'StyleGanGenerator':
        """
        Grow by a factor of 2 the generator's output resolution.
        :param (int) num_iters: number of iterations the new generator will run
        :param (str or torch.device or None) device: new network's device
        :return: a new StyleGanGenerator instance with doubled the resolution
        """
        self.freeze(force=True)
        # Init new Generator network
        new_resolution = 2 * self.resolution
        new_locals = self.locals.copy()
        new_locals['resolution'] = new_resolution
        new_locals['num_iters'] = num_iters
        new_locals['logger'] = self.logger
        new_gen = StyleGanGenerator(**new_locals)
        # Transfer parameter values
        new_gen.block0.load_state_dict(self.block0.state_dict())
        for bi in range(3, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            getattr(new_gen, f'upsample{block_index}') \
                .load_state_dict(getattr(self, f'upsample{block_index}').state_dict())
            getattr(new_gen, f'block{block_index}').load_state_dict(getattr(self, f'block{block_index}').state_dict())
            if bi == int(math.log2(self.resolution)):
                getattr(self, f'upsample{block_index}_toRGB') \
                    .load_state_dict(getattr(self, f'upsample{block_index}_toRGB').state_dict())
                getattr(self, f'block{block_index}_toRGB') \
                    .load_state_dict(getattr(self, f'block{block_index}_toRGB').state_dict())
        # Flush old network
        del self.constant
        del self.block0
        del self.block0_toRGB
        for bi in range(3, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            delattr(self, f'upsample{block_index}')
            delattr(self, f'block{block_index}')
            if bi == int(math.log2(self.resolution)):
                delattr(self, f'upsample{block_index}_toRGB')
                delattr(self, f'block{block_index}_toRGB')
        gc.collect()
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)
        # Return the initialized network
        return new_gen.to(device=device)


# noinspection DuplicatedCode
if __name__ == '__main__':
    # 4x4
    _stgen4 = StyleGanGenerator(resolution=4, truncation=1.0)
    # _noise = _stgen4.get_noise(1024, 'cpu')
    # fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # axs.hist(_noise.view(-1).numpy(), bins=100)
    # plt.show()
    _stdisc4 = StyleGanDiscriminator(resolution=4, c_in=3, adv_criterion='Wasserstein', use_gradient_penalty=True)
    print(to_human_readable(get_total_params(_stgen4)))
    enable_verbose(_stgen4)
    # enable_verbose(_stdisc4)
    # print(_stgen8)
    time.sleep(0.5)
    _gen_loss, _ = _stgen4.get_loss(batch_size=4, disc=_stdisc4)
    time.sleep(0.5)
    print(_gen_loss)

    # 8x8
    _stgen8 = _stgen4.grow()
    _stdisc8 = _stdisc4.grow()
    print(to_human_readable(get_total_params(_stgen8)))
    enable_verbose(_stgen8)
    # enable_verbose(_stdisc8)
    # print(_stgen8)
    time.sleep(0.5)
    _gen_loss, _ = _stgen8.get_loss(batch_size=4, disc=_stdisc8)
    time.sleep(0.5)
    print(_gen_loss)

    # 16x16
    _stgen16 = _stgen8.grow()
    _stdisc16 = _stdisc8.grow()
    print(to_human_readable(get_total_params(_stgen16)))
    enable_verbose(_stgen16)
    # enable_verbose(_stdisc16)
    time.sleep(0.5)
    _gen_loss, _ = _stgen16.get_loss(batch_size=4, disc=_stdisc16)
    time.sleep(0.5)
    print(_gen_loss)

    # 32x32
    _stgen32 = _stgen16.grow()
    _stdisc32 = _stdisc16.grow()
    print(to_human_readable(get_total_params(_stgen32)))
    enable_verbose(_stgen32)
    # enable_verbose(_stdisc32)
    time.sleep(0.5)
    _gen_loss, _ = _stgen32.get_loss(batch_size=4, disc=_stdisc32)
    time.sleep(0.5)
    print(_gen_loss)

    # 64x64
    _stgen64 = _stgen32.grow()
    _stdisc64 = _stdisc32.grow()
    print(to_human_readable(get_total_params(_stgen64)))
    enable_verbose(_stgen64)
    # enable_verbose(_stdisc64)
    time.sleep(0.5)
    _gen_loss, _ = _stgen64.get_loss(batch_size=4, disc=_stdisc64)
    time.sleep(0.5)
    print(_gen_loss)

    # 128x128
    _stgen128 = _stgen64.grow()
    _stdisc128 = _stdisc64.grow()
    print(to_human_readable(get_total_params(_stgen128)))
    enable_verbose(_stgen128)
    # enable_verbose(_stdisc128)
    time.sleep(0.5)
    _gen_loss, _ = _stgen128.get_loss(batch_size=4, disc=_stdisc128)
    time.sleep(0.5)
    print(_gen_loss)
