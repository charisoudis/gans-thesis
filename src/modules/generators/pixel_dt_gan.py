from typing import List, Optional, Tuple

import torch
from torch import nn

from modules.partial.decoding import ExpandingBlock
from modules.partial.encoding import ContractingBlock
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import enable_verbose


class PixelDTGanGenerator(nn.Module, BalancedFreezable, Verbosable):
    """
    PixelDTGANGenerator Class:
    This class implements the generator network from the PixelDTGAN paper ("Pixel-level Domain Transfer").
    """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 128, n_contracting_blocks: int = 5,
                 c_bottleneck: int = 64, w_in: int = 256, use_dropout: bool = True, use_out_tanh: bool = True,
                 logger: Optional[CommandLineLogger] = None):
        """
        PixelDTGANGenerator class constructor.
        :param (int) c_in: the number of channels to expect from a given input
        :param (int) c_out: the number of channels to expect for a given output
        :param (int) c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param (int) c_bottleneck: (G1) the number of channels to project down to before flattening for FC layer (this is
                             necessary since otherwise memory will be exhausted), in generator 1 network.
        :param (int) w_in: the input image width
        :param (bool) use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        :param (bool) use_out_tanh: set to True to use Tanh() activation in output layer; otherwise no output activation
                                    will be used
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        w_before_bottleneck = w_in // (2 ** (n_contracting_blocks - 1))
        self.gen = nn.Sequential(
            # Initial contracting block
            ContractingBlock(c_in=c_in, c_out=c_hidden, kernel_size=5, activation='lrelu', padding=2, use_norm=False),

            # Encoding (aka contracting) blocks
            *[ContractingBlock(c_hidden * 2 ** i, kernel_size=5, activation='lrelu', padding=2, use_norm=True,
                               use_dropout=(use_dropout and i < n_contracting_blocks // 2),
                               norm_type='batch') for i in range(0, n_contracting_blocks - 2)],

            # Bottleneck
            ContractingBlock(c_in=c_hidden * 2 ** (n_contracting_blocks - 2), c_out=c_bottleneck, stride=1, padding=0,
                             kernel_size=w_before_bottleneck, activation='lrelu', use_norm=True, norm_type='pixel'),
            ExpandingBlock(c_in=c_bottleneck, c_out=c_hidden * 2 ** (n_contracting_blocks - 2), padding=0,
                           output_padding=0, kernel_size=w_before_bottleneck, activation='relu',
                           use_norm=True, norm_type='pixel'),

            # Decoding (aka expanding) blocks
            *[ExpandingBlock(c_hidden * 2 ** i, kernel_size=5, activation='relu', use_norm=True, norm_type='batch',
                             padding=2, output_padding=1) for i in reversed(range(1, n_contracting_blocks - 1))],

            # Final expanding block
            ExpandingBlock(c_in=c_hidden, c_out=c_out, kernel_size=5, activation='tanh' if use_out_tanh else None,
                           padding=2, output_padding=1, use_norm=False),
        )

        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger
        self.verbose_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of PixelDTGanGenerator:
        Given an image tensor, passes it through the AE-like network and returns the output.
        :param x: image tensor of shape (N, 3(RGB) or 1(GS), H, W)
        :return: torch.Tensor of shape (N, c_out, H, W)
        """
        if self.verbose_enabled:
            self.logger.debug('_: ' + str(x.shape))
        return self.gen(x)

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, disc_r: nn.Module, disc_a: nn.Module,
                 adv_criterion: Optional[nn.modules.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the loss of the generator given inputs. If the criteria are not provided they will be set using the
        instance's given (or default) configuration.
        :param (torch.Tensor) x: input images (of people wearing the target garment)
        :param (torch.Tensor) y: target images (garments in neutral background)
        :param (torch.nn.Module) disc_r: the real/fake Discriminator network
        :param (torch.nn.Module) disc_a: the associated/unassociated Discriminator network
        :param (optional) adv_criterion: the adversarial loss function; takes the discriminator predictions and the
                                         target labels and returns a adversarial loss (which we aim to minimize)
        :return: a tuple containing the G1's loss (a scalar) and G2's loss (a scalar) and the outputs (for visualization
                 purposes)
        """
        # 1) Make a forward pass on the Generator
        y_hat = self(x)
        # 2) Compute Generator Loss
        #   - Adversarial loss for G2
        gen_out_predictions = disc_r(y_hat)
        adv_criterion = getattr(nn, f'{self.configuration["adv_criterion"]}Loss')() if not adv_criterion else \
            adv_criterion
        g2_loss_adv = adv_criterion(gen_out_predictions, torch.ones_like(gen_out_predictions))
        # - Aggregated loss
        g2_loss = g2_loss_adv + lambda_recon * g2_loss_recon
        return g1_loss, g2_loss, g1_out, g_out

    def get_layer_attr_names(self) -> List[str]:
        return ['gen', ]


if __name__ == '__main__':
    _gen = PixelDTGanGenerator(c_in=3, c_out=3, c_hidden=128, n_contracting_blocks=5, c_bottleneck=100, w_in=64,
                               use_dropout=True, use_out_tanh=True)
    # print(_gen)
    enable_verbose(_gen)
    _x = torch.randn(1, 3, 64, 64)
    _y = _gen(_x)
