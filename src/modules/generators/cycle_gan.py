from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import ExpandingBlock, FeatureMapLayer
from modules.partial.encoding import ContractingBlock
from modules.partial.residual import ResidualBlock


class CycleGANGenerator(nn.Module):
    """
     CycleGAN's Generator Class:
     A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to transform an input image into an
     image from the other class, with an upfeature layer at the start and a downfeature layer at the end.
     Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
     """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 64, n_residual_blocks: int = 9):
        """
        CycleGANGenerator class constructor.
        :param c_in: number of input channels
        :param c_out: number of output channels
        :param c_hidden: number of hidden channels (in residual blocks)
        :param n_residual_blocks: number of residual blocks
        """
        super(CycleGANGenerator, self).__init__()
        self.cycle_gan_generator = nn.Sequential(
            FeatureMapLayer(c_in, c_hidden),

            # Encoding (aka contracting) blocks (2)
            ContractingBlock(c_hidden),
            ContractingBlock(c_hidden * 2),

            # Residual Blocks
            *[ResidualBlock(c_hidden * 4, norm_type='IN') for _ in range(0, n_residual_blocks)],

            # Decoding (aka expanding) blocks (2)
            ExpandingBlock(c_hidden * 4, output_padding=1),
            ExpandingBlock(c_hidden * 2, output_padding=1),

            # Final layers
            FeatureMapLayer(c_hidden, c_out),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of Generator:
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        return self.cycle_gan_generator(x)

    ########################################
    # --------> Generator Losses <-------- #
    ########################################

    def get_adv_loss(self, real_x: Tensor, disc_y: nn.Module(), adv_criterion: nn.modules.Module = nn.MSELoss()) \
            -> Tuple[Tensor, Tensor]:
        """
        Return the adversarial loss of the generator given inputs (and the generated images for testing purposes).
        Attention: We suppose that this instance is the Generator from Domain X --> Y: gen_XY
        :param real_x: the real images from pile X
        :param disc_y: the discriminator for class Y; takes images and returns real/fake class Y prediction matrices
        :param adv_criterion: the adversarial loss function; takes the discriminator predictions and the target labels
                              and returns a adversarial loss (which we aim to minimize)
        :return: a tuple containing the loss (a scalar) and the outputs from generator's forward pass
        """
        fake_y = self(real_x)
        with torch.no_grad():
            fake_y_predictions = disc_y(fake_y)
        adversarial_loss = adv_criterion(fake_y_predictions, torch.ones_like(fake_y_predictions))
        return adversarial_loss, fake_y

    def get_identity_loss(self, real_y: Tensor, identity_criterion: nn.modules.Module = nn.L1Loss()) \
            -> Tuple[Tensor, Tensor]:
        """
        Return the identity loss of the generator given inputs (and the generated images for testing purposes).
        Attention: We suppose that this instance is the Generator from Domain X --> Y: gen_XY
        :param real_y: the real images from domain (or pile) Y
        :param identity_criterion: the identity loss function; takes the real images from Y and those images put
                                   through a X->Y generator and returns the identity loss (which we aim to minimize)
        :return: a tuple containing the loss (a scalar) and the outputs from generator's forward pass
        """
        identity_y = self(real_y)
        identity_loss = identity_criterion(identity_y, real_y)
        return identity_loss, identity_y

    def get_cycle_consistency_loss(self, real_y: Tensor, fake_x: Tensor,
                                   cycle_criterion: nn.modules.Module = nn.L1Loss()) -> Tuple[Tensor, Tensor]:
        """
        Return the cycle consistency loss of the generator given inputs (and the generated images for testing purposes).
        Attention: We suppose that this instance is the Generator from Domain X --> Y: gen_XY
        Parameters:
        :param real_y: the real images from domain (or pile) Y
        :param fake_x: the generated images of domain X (generated from gen_YX with input real_y)
        :param cycle_criterion: the cycle consistency loss function; takes the real images from Y and those images put
                                through a Y->X generator and then X->Y generator (this one) and returns the cycle
                                consistency loss (which we aim to minimize)
        :return: a tuple containing the loss (a scalar) and the outputs from generator's forward pass
        """
        cycle_y = self(fake_x)
        cycle_loss = cycle_criterion(cycle_y, real_y)
        return cycle_loss, cycle_y
