from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.partial.encoding import NoiseMappingNetwork
from modules.partial.normalization import AdaptiveInstanceNorm2d


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

    def __init__(self, c_in: int, c_out: int, w_dim: int = 512, kernel_size: int = 3,
                 upsample_size: Optional[int] = None):
        """
        StyleGanGeneratorBlock class constructor.
        :param (int) c_in: number of channels in the input
        :param (int) c_out: number of channels expected in the output
        :param (int) w_dim: size of the intermediate noise vector (defaults to 512)
        :param (int) kernel_size: nn.Conv2d()'s kernel_size argument (defaults to 3)
        :param (int or None) upsample_size: size of the nn.Upsample() output or None to disable upsampling the input
        """
        super().__init__()
        self.use_upsample = upsample_size is not None
        if self.use_upsample:
            self.upsample = nn.Upsample(size=upsample_size, mode='bilinear')

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
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv_block(x)
        x = self.adaptive_norm(x, w)
        return x


class StyleGanGenerator(nn.Module):
    """
    StyleGanGenerator Class.
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    """

    def __init__(self, z_dim, map_hidden_dim, w_dim, in_chan, out_chan, kernel_size, hidden_chan):
        """
        StyleGanGenerator class constructor.
        :param z_dim:
        :param map_hidden_dim:
        :param w_dim:
        :param in_chan:
        :param out_chan:
        :param kernel_size:
        :param hidden_chan:
        """
        super().__init__()
        self.map = NoiseMappingNetwork(z_dim, map_hidden_dim, w_dim)

        # Typically this constant is initiated to all ones, but you will initiate to a
        # Gaussian to better visualize the network's effect
        # Από εδώ (δλδ. από μια στεθερή τιμή) ξεκινάει ο Generator του StyleGAN
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))

        self.block0 = StyleGanGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, upsample_size=4)
        self.block1 = StyleGanGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, upsample_size=8)
        self.block2 = StyleGanGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, upsample_size=16)

        # You need to have a way of mapping from the output noise to an image,
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        # Στην ουσία είναι το τελευταίο block (output block)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        """
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger),
        upsamples the first to have the same dimensions as the second.
        Parameters:
        smaller_image: the smaller image to upsample
        bigger_image: the bigger image whose dimensions will be upsampled to
        """
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        """
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
        computes a StyleGAN iteration.
        Parameters:
        noise: a noise tensor with dimensions (n_samples, z_dim)
        return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        """
        x = self.starting_constant
        w = self.map(noise)  # --> (N, w_dim)
        x = self.block0(x, w)  # --> (N, C_hidden, 4x4)
        x_small = self.block1(x, w)  # --> (N, C_hidden, 8x8) - First generator run output
        x_small_image = self.block1_to_image(x_small)  # --> (N, C_out, 8x8)
        x_big = self.block2(x_small, w)  # --> (N, C_hidden, 16x16) - Second generator run output
        x_big_image = self.block2_to_image(x_big)  # --> (N, C_out, 16x16)

        # Upsample first generator run output to be same size as second generator run output
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)
        # Interpolate between the upsampled image and the image from the generator using alpha
        interpolation = (1 - self.alpha) * x_small_upsample + self.alpha * x_big_image

        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation
