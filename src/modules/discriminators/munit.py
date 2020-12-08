from typing import List

import torch.nn as nn
from torch import Tensor

from modules.discriminators.patch_gan import PatchGANDiscriminator


class MUNITDiscriminator(nn.Module):
    """
    MUNITDiscriminator Class:
    MUNIT, like Pix2PixHD, uses multiple PatchGAN discriminators at multiple downsampled versions of the input image.
    """

    def __init__(self, n_discriminators: int = 3, c_in: int = 3, c_hidden: int = 64, n_contracting_blocks: int = 3):
        """
        MUNITDiscriminator class constructor.
        :param n_discriminators: number PatchGAN discriminators used
        :param c_in: number of input channels after first Conv2d layer
        :param c_hidden: number of hidden channels in PatchGAN discriminator
        :param n_contracting_blocks: number of contracting blocks
        """
        super(MUNITDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(c_in, c_hidden, n_contracting_blocks, use_spectral_norm=True)
            for _ in range(n_discriminators)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Function for completing a forward pass of MUNITDiscriminator:
        Given an image tensor, returns a list of 2D matrices of realness probabilities for each image's "patches" and
        each downsampled version.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: list of PatchGAN Discriminators' outputs
        """
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)
        return outputs

    def get_loss(self, real: Tensor, fake: Tensor, criterion: nn.Module = nn.BCELoss) -> Tensor:
        # TODO
        pass
