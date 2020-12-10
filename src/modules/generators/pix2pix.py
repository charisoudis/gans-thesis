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

    def get_loss(self, disc: nn.Module, real: Tensor, condition: Tensor,
                 adv_criterion: nn.modules.Module = nn.BCEWithLogitsLoss(),
                 recon_criterion: nn.modules.Module = nn.L1Loss(), lambda_recon: int = 10):
        """
        Return the loss of the generator given inputs. This Generator takes the condition image as input and returns
        a generated potential image (for every condition image in batch).
        :param disc: the discriminator; takes images and the condition and returns real/fake prediction matrices
        :param real: the real images (e.g. maps) to be used to evaluate the reconstruction
        :param condition: the source images (e.g. satellite imagery) which are used to pro
        :param adv_criterion: the adversarial loss function; takes the discriminator predictions and the true labels
        and returns a adversarial loss (which you aim to minimize)
        :param recon_criterion: the reconstruction loss function; takes the generator outputs and the real images and
        returns a reconstruction loss (which we aim to minimize)
        :param lambda_recon: the degree to which the reconstruction loss should be weighted
        """
        fake_images = self(condition)
        fake_predictions = disc(fake_images, condition)
        recon_loss = recon_criterion(fake_images, real)
        if type(adv_criterion) == torch.nn.modules.loss.BCELoss:
            fake_predictions = nn.Sigmoid()(fake_predictions)
        adv_loss = adv_criterion(fake_predictions, torch.ones_like(fake_predictions))
        return adv_loss + lambda_recon * recon_loss
