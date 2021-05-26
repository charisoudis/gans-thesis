from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import ExpandingBlock, FeatureMapLayer
from modules.partial.encoding import ContractingBlock
from modules.partial.residual import ResidualBlock
from utils.ifaces import BalancedFreezable
from utils.pytorch import enable_verbose, get_total_params


class CycleGANGenerator(nn.Module, BalancedFreezable):
    """
     CycleGAN's Generator Class:
     A series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks to transform an input image into an
     image from the other class, with an upfeature layer at the start and a downfeature layer at the end.
     Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
     """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 64, n_contracting_blocks: int = 2,
                 n_residual_blocks: int = 9, output_padding: int = 1):
        """
        CycleGANGenerator class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_out: number of output channels
        :param (int) c_hidden: number of hidden channels (in residual blocks)
        :param (int) n_contracting_blocks: number of contracting (and expanding) blocks
        :param (int) n_residual_blocks: number of residual blocks
        :param (int) output_padding: :attr:`output_padding` of `nn.Conv2d()`
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        self.upfeature = FeatureMapLayer(c_in, c_hidden)
        self.n_contracting_blocks = n_contracting_blocks

        # Register contracting blocks
        for i in range(n_contracting_blocks):
            setattr(self, f'contract{(i + 1)}', ContractingBlock(c_in=c_hidden * 2 ** i))

        # Register residual blocks
        c_res = c_hidden * 2 ** n_contracting_blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(c_res, norm_type='IN') for _ in range(0, n_residual_blocks)]
        )

        # Register expanding blocks
        for i in range(n_contracting_blocks):
            setattr(self, f'expand{i}', ExpandingBlock(c_res // 2 ** i, output_padding=output_padding, use_skip=True))

        # Register output block
        self.out = nn.Sequential(
            FeatureMapLayer(c_hidden, c_out),
            nn.Tanh()
        )
        # self.cycle_gan_generator = nn.Sequential(
        #     FeatureMapLayer(c_in, c_hidden),
        #
        #     # Encoding (aka contracting) blocks (2)
        #     ContractingBlock(c_hidden),
        #     ContractingBlock(c_hidden * 2),
        #
        #     # Residual Blocks
        #     *[ResidualBlock(c_hidden * 4, norm_type='IN') for _ in range(0, n_residual_blocks)],
        #
        #     # Decoding (aka expanding) blocks (2)
        #     ExpandingBlock(c_hidden * 4, output_padding=1),
        #     ExpandingBlock(c_hidden * 2, output_padding=1),
        #
        #     # Final layers
        #     FeatureMapLayer(c_hidden, c_out),
        #     nn.Tanh()
        # )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of Generator:
        Given an image tensor, passes it through the U-Net with residual blocks
        and returns the output.
        :param (torch.Tensor) x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        x0 = self.upfeature(x)

        # Pass through contracting blocks
        out = None
        contracting_block_outs = [x0]
        for i in range(self.n_contracting_blocks):
            contracting_block = getattr(self, f'contract{(i + 1)}')
            out = contracting_block(contracting_block_outs[-1])
            contracting_block_outs.append(out)

        # Pass through residual blocks
        out = self.res_blocks(out)

        # Pass through expanding blocks
        for i in range(self.n_contracting_blocks):
            expanding_block = getattr(self, f'expand{i}')
            out = expanding_block(out, contracting_block_outs[self.n_contracting_blocks - (i + 1)])

        return self.out(out)
        # return self.cycle_gan_generator(x)

    ########################################
    # --------> Generator Losses <-------- #
    ########################################

    def get_adv_loss(self, real_x: Tensor, disc_y: nn.Module(), adv_criterion: nn.modules.Module = nn.MSELoss()) \
            -> Tuple[Tensor, Tensor]:
        """
        Return the adversarial loss of the generator given inputs (and the generated images for testing purposes).
        Attention: We suppose that this instance is the Generator from Domain X --> Y: gen_XY
        :param (torch.Tensor) real_x: the real images from pile X
        :param (torch.nn.Module) disc_y: the discriminator for class Y; takes images and returns real/fake class Y
                                         prediction matrices
        :param (torch.nn.Module) adv_criterion: the adversarial loss function; takes the discriminator predictions and
                                                the target labels and returns a adversarial loss (which we aim to
                                                minimize)
        :return: a tuple containing the loss (a scalar) and the outputs from generator's forward pass
        """
        fake_y = self(real_x)
        fake_y_predictions = disc_y(fake_y)
        adversarial_loss = adv_criterion(fake_y_predictions, torch.ones_like(fake_y_predictions))
        return adversarial_loss, fake_y

    def get_identity_loss(self, real_y: Tensor, identity_criterion: nn.modules.Module = nn.L1Loss()) \
            -> Tuple[Tensor, Tensor]:
        """
        Return the identity loss of the generator given inputs (and the generated images for testing purposes).
        Attention: We suppose that this instance is the Generator from Domain X --> Y: gen_XY
        :param (torch.Tensor) real_y: the real images from domain (or pile) Y
        :param (torch.nn.Module) identity_criterion: the identity loss function; takes the real images from Y and those
                                                     images put through a X->Y generator and returns the identity loss
                                                     (which we aim to minimize)
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
        :param (torch.Tensor) real_y: the real images from domain (or pile) Y
        :param (torch.Tensor) fake_x: the generated images of domain X (generated from gen_YX with input real_y)
        :param (torch.nn.Module) cycle_criterion: the cycle consistency loss function; takes the real images from Y and
                                                  those images put through a Y->X generator and then X->Y generator
                                                  (this one) and returns the cycle consistency loss (which we aim to
                                                  minimize)
        :return: a tuple containing the loss (a scalar) and the outputs from generator's forward pass
        """
        cycle_y = self(fake_x)
        cycle_loss = cycle_criterion(cycle_y, real_y)
        return cycle_loss, cycle_y


if __name__ == '__main__':
    _cycle_gan = CycleGANGenerator(c_in=3, c_out=3)
    enable_verbose(_cycle_gan)
    _x = torch.rand(1, 3, 64, 64)
    _cycle_gan(_x)

    get_total_params(_cycle_gan, print_table=True, sort_desc=True)
