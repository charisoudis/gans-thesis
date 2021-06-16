import math
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.encoding import NoiseMappingNetwork
from modules.partial.normalization import AdaptiveInstanceNorm2d
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable
from utils.pytorch import get_total_params, enable_verbose
from utils.string import to_human_readable


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
            nn.LeakyReLU(0.2, inplace=True)
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


class StyleGanGenerator(nn.Module, BalancedFreezable):
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
                 kernel_size: int = 3, alpha_init: float = 1e-2, logger: Optional[CommandLineLogger] = None):
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
        :param (float) alpha_init: initial mixing weight used for progressing growing (defaults to 1e-2)
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
        for bi in range(3, int(math.log2(resolution)) + 1):
            block_index = bi - 2
            setattr(self, f'upsample{block_index}', nn.Upsample(size=2 ** bi, mode='bilinear', align_corners=False))
            setattr(self, f'block{block_index}',
                    StyleGanGeneratorBlock(c_in=c_hidden, c_out=c_hidden, w_dim=w_dim, kernel_size=kernel_size))
            if bi == int(math.log2(resolution)):
                setattr(self, f'project{block_index}_upsample', nn.Conv2d(c_hidden, c_out, kernel_size=1))
                setattr(self, f'project{block_index}_block', nn.Conv2d(c_hidden, c_out, kernel_size=1))
        # Save args
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.resolution = self.locals['resolution']

        self.logger = logger if logger is not None else \
            CommandLineLogger(log_level='debug', name=self.__class__.__name__)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        :param (torch.tensor) z: input noise vector of shape (N, z_dim)
        :return: a torch.Tensor object containing the mixing of the output of the UpSampler and the last conv block
        """
        # Map noise
        w = self.noise_mapping(z)  # (N, w_dim)
        # Pass through generator blocks
        #   - constant + first block
        x = self.constant
        x = self.block0(x, w)
        #   - upsampling + conv blocks until the last (wherein the mixing happens)
        for bi in range(3, int(math.log2(self.resolution)) + 1):
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
                return (1.0 - self.alpha) * x_upsampled + self.alpha * x
            x = block(x, w)
        return x

    def grow(self) -> 'StyleGanGenerator':
        """
        Grow by a factor of 2 the generator's output resolution.
        :return: a new StyleGanGenerator instance with doubled the resolution
        """
        # Init new Generator network
        new_resolution = 2 * self.resolution
        new_locals = self.locals.copy()
        new_locals['resolution'] = new_resolution
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
        return new_gen.to(self.alpha.device)


if __name__ == '__main__':
    _z = torch.randn(1, 512)

    # 8x8
    _stgen8 = StyleGanGenerator(resolution=8)
    print(to_human_readable(get_total_params(_stgen8)))
    enable_verbose(_stgen8)
    # print(_stgen8)
    time.sleep(0.5)
    _y = _stgen8(_z)
    time.sleep(0.5)
    print(_y.shape)

    # 16x16
    _stgen16 = _stgen8.grow()
    print(to_human_readable(get_total_params(_stgen16)))
    enable_verbose(_stgen16)
    time.sleep(0.5)
    _y = _stgen16(_z)
    time.sleep(0.5)
    print(_y.shape)

    # 32x32
    _stgen32 = _stgen16.grow()
    print(to_human_readable(get_total_params(_stgen32)))
    enable_verbose(_stgen32)
    time.sleep(0.5)
    _y = _stgen32(_z)
    time.sleep(0.5)
    print(_y.shape)

    # 64x64
    _stgen64 = _stgen32.grow()
    print(to_human_readable(get_total_params(_stgen64)))
    enable_verbose(_stgen64)
    time.sleep(0.5)
    _y = _stgen64(_z)
    time.sleep(0.5)
    print(_y.shape)

    # 128x128
    _stgen128 = _stgen64.grow()
    print(to_human_readable(get_total_params(_stgen128)))
    enable_verbose(_stgen128)
    time.sleep(0.5)
    _y = _stgen128(_z)
    time.sleep(0.5)
    print(_y.shape)
