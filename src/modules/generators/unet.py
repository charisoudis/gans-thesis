from typing import Optional

import torch
from torch import nn, Tensor

from modules.partial.decoding import UNETExpandingBlock, FeatureMapLayer, ChannelsProjectLayer
from modules.partial.encoding import UNETContractingBlock
from utils.ifaces import BalancedFreezable, Freezable
from utils.pytorch import enable_verbose


class UNETWithSkipConnections(nn.Module, BalancedFreezable):
    """
    UNETWithSkipConnections Class:
    A series of 4 contracting blocks followed by 4 expanding blocks to transform an input image into the corresponding
    paired image, with an upfeature layer at the start and a downfeature layer at the end. Unlike UNET paper, this is a
    "symmetric" implementation, which means that (H,W)_out = (H,W)_in. This is necessary of image-to-image translation
    tasks where pix2pix is used.
    """

    def __init__(self, c_in, c_out, c_hidden=32, n_contracting_blocks: int = 4, use_dropout: bool = False,
                 use_bn: bool = False, fc_in_bottleneck: bool = False, h_in: Optional[int] = None,
                 w_in: Optional[int] = None, c_bottleneck_down: int = 10, use_out_tanh: bool = True):
        """
        UNETWithSkipConnections class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param n_contracting_blocks: the base number of contracting (and corresponding expanding) blocks
        :param use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        :param use_bn: set to True to use Batch Normalization after every convolution layer
        :param fc_in_bottleneck: set to True to apply a FC layer in the bottleneck (e.g. for PGPG this is set to True)
        :param h_in: required if fc_in_bottleneck is True, to calculate FC layer size
        :param w_in: required if fc_in_bottleneck is True, to calculate FC layer size
        :param c_bottleneck_down: the number of channels to project down to before flattening for FC layer (this is
                                  necessary since otherwise memory will be exhausted)
        :param use_out_tanh: set to True to use Tanh() activation in output layer; otherwise no output activation will
                             be used
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        self.upfeature = FeatureMapLayer(c_in, c_hidden)
        self.n_contracting_blocks = n_contracting_blocks

        # Contracting blocks
        for i in range(n_contracting_blocks):
            setattr(self, f'contract{(i + 1)}',
                    UNETContractingBlock(c_hidden * 2 ** i, use_bn=use_bn,
                                         use_dropout=use_dropout and i < n_contracting_blocks // 2))

        # Bottleneck FC
        self.c_bottleneck = c_hidden * 2 ** n_contracting_blocks
        self.fc_in_bottleneck = fc_in_bottleneck
        if self.fc_in_bottleneck:
            if w_in is None or h_in is None:
                raise ValueError('When fc_in_bottleneck=True, (h_in, w_in) are required')
            w_bottleneck = w_in // 2 ** n_contracting_blocks
            h_bottleneck = h_in // 2 ** n_contracting_blocks
            bottleneck_neurons_count = c_bottleneck_down * w_bottleneck * h_bottleneck
            self.bottleneck = nn.Sequential(
                ChannelsProjectLayer(self.c_bottleneck, c_bottleneck_down),
                nn.Flatten(),
                nn.Linear(bottleneck_neurons_count, bottleneck_neurons_count),
                # nn.LeakyReLU(),
                nn.Unflatten(1, (c_bottleneck_down, h_bottleneck, w_bottleneck)),
                ChannelsProjectLayer(c_bottleneck_down, self.c_bottleneck),
            )

        # Expanding blocks (symmetric)
        for i in range(n_contracting_blocks):
            setattr(self, f'expand{i}', UNETExpandingBlock(self.c_bottleneck // 2 ** i))

        # Output block
        self.out = nn.Sequential(
            FeatureMapLayer(c_hidden, c_out),
            torch.nn.Tanh()
        ) if use_out_tanh else FeatureMapLayer(c_hidden, c_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of UNETWithSkipConnections:
        Given an image tensor, passes it through U-Net and returns the output.
        :param x: image tensor of shape (N, 3(RGB) or 1(GS), H, W)
        :return: torch.Tensor of shape (N, <number or classes>, H, W)
        """
        x0 = self.upfeature(x)

        # Pass through contracting blocks
        out = None
        contracting_block_outs = [x0]
        for i in range(self.n_contracting_blocks):
            contracting_block = getattr(self, f'contract{(i + 1)}')
            out = contracting_block(contracting_block_outs[-1])
            contracting_block_outs.append(out)

        # Apply fully connected in bottleneck
        if self.fc_in_bottleneck:
            out = self.bottleneck(out)

        # Pass through expanding blocks
        for i in range(self.n_contracting_blocks):
            expanding_block = getattr(self, f'expand{i}')
            out = expanding_block(out, contracting_block_outs[self.n_contracting_blocks - (i + 1)])

        return self.out(out)

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
        # 1) Freeze Discriminator
        assert isinstance(disc, Freezable), 'Discriminator must implement utils.ifaces.Freezable in order to be frozen'
        with disc.frozen():
            # 2) Forward pass through UNET
            fake_images = self(condition)
            fake_predictions = disc(fake_images, condition)
            recon_loss = recon_criterion(fake_images, real)
            if type(adv_criterion) == torch.nn.modules.loss.BCELoss:
                fake_predictions = nn.Sigmoid()(fake_predictions)
            adv_loss = adv_criterion(fake_predictions, torch.ones_like(fake_predictions))
        return adv_loss + lambda_recon * recon_loss


if __name__ == '__main__':
    _unet = UNETWithSkipConnections(c_in=3, c_out=3)
    enable_verbose(_unet)
    _x = torch.rand(1, 3, 64, 64)
    _unet(_x)
