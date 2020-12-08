from typing import List

import torch
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
        :param n_contracting_blocks: number of contracting blocks PatchGAN discriminator
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

    def get_loss(self, images: Tensor, is_real: bool = True, criterion: nn.Module = nn.MSELoss) -> Tensor:
        """
        Compute adversarial loss.
        :param images: image tensor (batch of images) to be classified for its realness
        :param is_real: whether the batch comes from the real dataset or from the Generator
        :param criterion: cost function to be used (in MUNIT paper the use LSGAN and thus MSE is the adversarial loss)
        :return:
        """
        batch_predictions_on_images = self(images)
        loss = torch.Tensor(0.0)
        for batch in batch_predictions_on_images:
            loss += criterion(batch, torch.ones_like(batch) if is_real else torch.zeros_like(batch))
        return loss
