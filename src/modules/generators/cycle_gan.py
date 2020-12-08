import torch.nn as nn
from torch import Tensor
from modules.partial.encoding import ContractingBlock
from modules.partial.residual import ResidualBlock
from modules.partial.decoding import ExpandingBlock, FeatureMapLayer


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
            *[ResidualBlock(c_hidden * 4) for _ in range(0, n_residual_blocks)],

            # Decoding (aka expanding) blocks (2)
            ExpandingBlock(c_hidden * 4),
            ExpandingBlock(c_hidden * 2),

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
