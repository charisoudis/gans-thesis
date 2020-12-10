import torch
import torch.nn as nn
from torch import Tensor
from modules.partial.encoding import ContractingBlock
from modules.partial.decoding import FeatureMapLayer, ChannelsProjectLayer


class PatchGANDiscriminator(nn.Module):
    """
    PatchGANDiscriminator Class:
    Outputs a map of real/fake probabilities.
    Parameters:
        c_in: the number of image input channels
        c_hidden: the initial number of discriminator convolutional filters
    """

    def __init__(self, c_in: int, c_hidden: int = 8, n_contracting_blocks: int = 4, use_spectral_norm: bool = False):
        """
        PatchGANDiscriminator class constructor.
        :param c_in: number of input channels
        :param c_hidden: the initial number of discriminator convolutional filters (channels)
        :param n_contracting_blocks: number of contracting blocks
        :param use_spectral_norm: flag to use/not use Spectral Normalization in ChannelsProject layer
        """
        super(PatchGANDiscriminator, self).__init__()
        self.patch_gan_discriminator = nn.Sequential(
            FeatureMapLayer(c_in, c_hidden),

            # Encoding (aka contracting) blocks
            ContractingBlock(c_hidden, use_bn=False),
            *[ContractingBlock(c_hidden * 2 ** i) for i in range(1, n_contracting_blocks)],

            ChannelsProjectLayer(c_hidden * 2 ** n_contracting_blocks, 1, use_spectral_norm=use_spectral_norm)
        )

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        """
        Function for completing a forward pass of PatchGANDiscriminator:
        Given an image tensor, returns a 2D matrix of realness probabilities for each image's "patches".
        :param x: image tensor of shape (N, C_in, H, W)
        :param y: image tensor of shape (N, C_y, H, W) containing the condition images (e.g. for pix2pix)
        :return: transformed image tensor of shape (N, 1, P_h, P_w)
        """
        if y is not None:
            x = torch.cat([x, y], dim=1)  # channel-wise concatenation
        return self.patch_gan_discriminator(x)

    def get_loss(self, real: Tensor, fake: Tensor, condition: Tensor = None,
                 criterion: nn.modules.Module = nn.BCELoss()) -> Tensor:
        """
        Compute adversarial loss.
        :param real: image tensor of shape (N, C, H, W) from real dataset
        :param fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param condition: condition image tensor of shape (N, C_in/2, H, W) that is stacked before input to PatchGAN
        discriminator
        :param criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        predictions_on_real = self(real, condition)
        predictions_on_fake = self(fake, condition)
        if type(criterion) == torch.nn.modules.loss.BCELoss:
            predictions_on_real = nn.Sigmoid()(predictions_on_real)
            predictions_on_fake = nn.Sigmoid()(predictions_on_fake)
        loss_on_real = criterion(predictions_on_real, torch.ones_like(predictions_on_real))
        loss_on_fake = criterion(predictions_on_fake, torch.zeros_like(predictions_on_real))
        return 0.5 * (loss_on_real + loss_on_fake)
