import math
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
        noise_shape = (batch_size, 1, h, w)
        noise = torch.randn(noise_shape, device=x.device)
        return x + self.weight * noise


class StyleGanGeneratorBlock(nn.Module):
    """
    StyleGanGeneratorBlock Class.
    This class implements a single block of the StyleGAN v1 generator.
    """

    def __init__(self, c_in: int, c_out: int, w_dim: int = 512, kernel_size: int = 3):
        """
        StyleGanGeneratorBlock class constructor.
        :param (int) c_in: number of channels in the input
        :param (int) c_out: number of channels expected in the output
        :param (int) w_dim: size of the intermediate noise vector (defaults to 512)
        :param (int) kernel_size: nn.Conv2d()'s kernel_size argument (defaults to 3)
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=1),
            InjectNoise(c_out),
            nn.LeakyReLU(0.2)
        )
        self.adaptive_norm = AdaptiveInstanceNorm2d(c_out, w_dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        Function for completing a forward pass of StyleGanGeneratorBlock:
        Given an input x and the style vector w, it computes a StyleGAN generator block.
        :param (torch.Tensor) x: the input to the generator's block, of shape (N,C_in,H,W)
        :param (torch.Tensor) w: the intermediate noise vector, of shape (N, W_dim)
        :return a torch.Tensor object of shape (N, C_out, H, W)
        """
        x = self.conv_block(x)
        return self.adaptive_norm(x, w)


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
                 c_const: int = 512, c_hidden: int = 1024, c_out: int = 3, resolution: int = 128,
                 kernel_size: int = 3, alpha_multiplier: float = 10.0, num_iters: int = 1,
                 logger: Optional[CommandLineLogger] = None):
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
        """
        # Save arguments
        self.locals = locals()
        del self.locals['self']

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
        self.block0 = StyleGanGeneratorBlock(c_in=c_const, c_out=c_hidden, w_dim=w_dim, kernel_size=kernel_size)
        self.project0_block = nn.Conv2d(c_hidden, c_out, kernel_size=1)
        for bi in range(3, int(math.log2(resolution)) + 1):
            block_index = bi - 2
            setattr(self, f'upsample{block_index}', nn.Upsample(size=2 ** bi, mode='bilinear', align_corners=False))
            setattr(self, f'block{block_index}',
                    StyleGanGeneratorBlock(c_in=c_hidden, c_out=c_hidden, w_dim=w_dim, kernel_size=kernel_size))
            if bi == int(math.log2(resolution)):
                setattr(self, f'project{block_index}_upsample', nn.Conv2d(c_hidden, c_out, kernel_size=1))
                setattr(self, f'project{block_index}_block', nn.Conv2d(c_hidden, c_out, kernel_size=1))
        # Save args
        self.resolution = self.locals['resolution']

        # Fade coefficient
        self.alpha_curve = get_alpha_curve(num_iters=num_iters, alpha_multiplier=alpha_multiplier)
        self.alpha_index = 0
        self.alpha = self.alpha_curve[self.alpha_index]

        self.logger = logger if logger is not None else \
            CommandLineLogger(log_level='debug', name=self.__class__.__name__)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        :param (torch.tensor) z: input noise vector of shape (N, z_dim)
        :return: a torch.Tensor object containing the mixing of the output of the UpSampler and the last conv block
        """
        # Get current alpha value
        alpha = self.alpha
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
        w = self.noise_mapping(z)  # (N, w_dim)
        # Pass through generator blocks
        #   - constant + first block
        x = self.constant
        x = self.block0(x, w)
        if resolution_log2 == 2:
            return self.project0_block(x)
        #   - upsampling + conv blocks until the last (wherein the mixing happens)
        for bi in range(3, resolution_log2 + 1):
            block_index = bi - 2
            upsample = getattr(self, f'upsample{block_index}')
            x = upsample(x)
            block = getattr(self, f'block{block_index}')
            if bi == int(math.log2(self.resolution)):
                project_upsample = getattr(self, f'project{block_index}_upsample')
                project_block = getattr(self, f'project{block_index}_block')
                x_upsampled = project_upsample(x)
                x = block(x, w)
                x = project_block(x)
                return x if alpha >= 1.0 else (1.0 - alpha) * x_upsampled + alpha * x
            x = block(x, w)
        return x

    def get_noise(self, batch_size: int, device: Optional[str or torch.device] = None) -> torch.Tensor:
        """
        Create and return a new noise vector in the same device as the model itself.
        :param (int) batch_size: number of vectors in batch
        :param (str or torch.device or None) device: vector's device
        :return: a torch.Tensor object of shape (N, z_dim)
        """
        return torch.randn(batch_size, self.locals['z_dim'], device=device)

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
        layer_names = ['block0', 'project0_block']
        for bi in range(3, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            layer_names.append(f'upsample{block_index}')
            layer_names.append(f'block{block_index}')
            if bi == int(math.log2(self.resolution)):
                layer_names.append(f'project{block_index}_upsample')
                layer_names.append(f'project{block_index}_block')
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
                getattr(self, f'project{block_index}_upsample') \
                    .load_state_dict(getattr(self, f'project{block_index}_upsample').state_dict())
                getattr(self, f'project{block_index}_block') \
                    .load_state_dict(getattr(self, f'project{block_index}_block').state_dict())
        # Return the initialized network
        return new_gen.to(device=device)


# noinspection DuplicatedCode
if __name__ == '__main__':
    # 4x4
    _stgen4 = StyleGanGenerator(resolution=4)
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
