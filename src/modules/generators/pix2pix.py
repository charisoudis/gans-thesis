import torch
from torch import nn, Tensor

from modules.partial.encoding import UNETContractingBlock
from modules.partial.decoding import UNETExpandingBlock, FeatureMapLayer


class UNet(nn.Module):
    """
    UNet Class:
    A series of 4 contracting blocks followed by 4 expanding blocks to transform an input image into the corresponding
    paired image, with an upfeature layer at the start and a downfeature layer at the end. Unlike UNET paper, this is a
    "symmetric" implementation, which means that (H,W)_out = (H,W)_in. This is necessary of image-to-image translation
    tasks where pix2pix is used.
    """

    def __init__(self, c_in, c_out, c_hidden=32):
        """
        UNet class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        """
        super(UNet, self).__init__()
        self.upfeature = FeatureMapLayer(c_in, c_hidden)
        self.contract1 = UNETContractingBlock(c_hidden, use_dropout=True)
        self.contract2 = UNETContractingBlock(c_hidden * 2, use_dropout=True)
        self.contract3 = UNETContractingBlock(c_hidden * 4, use_dropout=True)
        self.contract4 = UNETContractingBlock(c_hidden * 8)
        self.contract5 = UNETContractingBlock(c_hidden * 16)
        self.contract6 = UNETContractingBlock(c_hidden * 32)
        self.expand0 = UNETExpandingBlock(c_hidden * 64)
        self.expand1 = UNETExpandingBlock(c_hidden * 32)
        self.expand2 = UNETExpandingBlock(c_hidden * 16)
        self.expand3 = UNETExpandingBlock(c_hidden * 8)
        self.expand4 = UNETExpandingBlock(c_hidden * 4)
        self.expand5 = UNETExpandingBlock(c_hidden * 2)
        self.downfeature = FeatureMapLayer(c_hidden, c_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        :param x: image tensor of shape (N, 3(RGB) or 1(GS), H, W)
        :return: torch.Tensor of shape (N, <number or classes>, H, W)
        """
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.sigmoid(xn)
