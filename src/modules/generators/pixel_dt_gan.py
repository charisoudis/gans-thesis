from typing import List, Optional, Tuple, Dict

import torch
from torch import nn
from torch.nn import L1Loss

from modules.partial.decoding import ExpandingBlock
from modules.partial.encoding import ContractingBlock
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import enable_verbose, get_total_params


class PixelDTGanGenerator(nn.Module, BalancedFreezable, Verbosable):
    """
    PixelDTGANGenerator Class:
    This class implements the generator network from the PixelDTGAN paper ("Pixel-level Domain Transfer").
    """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 128, n_contracting_blocks: int = 5,
                 c_bottleneck: int = 64, w_in: int = 256, use_dropout: bool = True, use_out_tanh: bool = True,
                 adv_criterion_conf: Optional[dict] = None, logger: Optional[CommandLineLogger] = None):
        """
        PixelDTGANGenerator class constructor.
        :param (int) c_in: the number of channels to expect from a given input
        :param (int) c_out: the number of channels to expect for a given output
        :param (int) c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param (int) c_bottleneck: number of channels/elements in the bottleneck layer (c_bottleneck x 1 x 1)
        :param (int) w_in: the input image width
        :param (bool) use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        :param (bool) use_out_tanh: set to True to use Tanh() activation in output layer; otherwise no output activation
                                    will be used
        :param (optional) adv_criterion_conf: the configuration parameters of the network. Among other parameters the
                                              following keys are necessary to instantiate the model:
                                                - real: type of Adversarial loss for real/fake discriminator
                                                predictions. Supported: `nn.MSELoss()`, `nn.BCELoss()`,
                                                `nn.BCEWithLogitsLoss()`
                                                - associated: Adversarial loss for associated/unassociated
                                                discriminator's predictions. Supported: `nn.MSELoss()`, `nn.BCELoss()`,
                                                `nn.BCEWithLogitsLoss()`
                                                , logger: Optional[CommandLineLogger] = None
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
                               norm_type='batch') for i in range(n_contracting_blocks - 2)],

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

        # Save arguments
        self.adv_criterion_conf = adv_criterion_conf
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger
        self.verbose_enabled = False

        # Initiate reconstruction weight
        self.lambda_recon = nn.Parameter(torch.tensor(4.0, requires_grad=True))

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        Overrides `torch.nn.Module.load_state_dict()`.
        :param state_dict: see `torch.nn.Module.load_state_dict()`
        :param strict: see `torch.nn.Module.load_state_dict()`
        :return: see `torch.nn.Module.load_state_dict()`
        """
        # Check for lambda_recon presence in state
        if 'lambda_recon' not in state_dict.keys():
            state_dict['lambda_recon'] = self.lambda_recon.data
        # FIX: Update keys for expanding block
        state_dict = ExpandingBlock.fix_state_dict(state_dict)
        # Load model state
        # noinspection PyTypeChecker
        nn.Module.load_state_dict(self, state_dict=state_dict, strict=strict)

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

    def get_loss(self, img_s: torch.Tensor, img_t: torch.Tensor, disc_r: nn.Module, disc_a: nn.Module,
                 adv_criterion: Optional[nn.modules.Module] = None,
                 recon_criterion: nn.Module = L1Loss()) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the loss of the generator given inputs. If the criteria are not provided they will be set using the
        instance's given (or default) configuration.
        :param (torch.Tensor) img_s: input images (of people wearing the target garments)
        :param (torch.Tensor) img_t: target output images (of the target garments)
        :param (torch.nn.Module) disc_r: the real/fake Discriminator network
        :param (torch.nn.Module) disc_a: the associated/unassociated Discriminator network
        :param (optional) adv_criterion: the adversarial loss function; takes the discriminator predictions and the
                                         target labels and returns a adversarial loss (which we aim to minimize)
        :param (nn.Module) recon_criterion: the reconstruction loss function; takes the generator outputs and the real
                                            ones and returns a reconstruction loss (which we aim to minimize).
                                            Defaults to L1 loss.
        :return: a tuple containing the aggregated Generator's loss (a scalar) and the output batch of images (for
                 visualization purposes)
        """
        # 1) Make a forward pass on the Generator
        img_t_hat = self(img_s)
        # 2) Compute Generator Loss
        #   - Adversarial loss from Real/Fake Discriminator
        disc_r_predictions = disc_r(img_t_hat.clone())
        adv_criterion_r = self.adv_criterion_conf['real'] if not adv_criterion else adv_criterion
        adv_loss_r = adv_criterion_r(disc_r_predictions, torch.ones_like(disc_r_predictions))
        #   - Adversarial loss from Associated/Unassociated Discriminator
        disc_a_predictions = disc_a(img_t_hat, img_s)
        adv_criterion_a = self.adv_criterion_conf['associated'] if not adv_criterion else adv_criterion
        adv_loss_a = adv_criterion_a(disc_a_predictions, torch.ones_like(disc_a_predictions))
        #   - Reconstruction loss
        recon_loss = recon_criterion(img_t_hat, img_t)
        #   - Aggregate losses (mean)
        gen_loss = torch.mean(torch.stack([adv_loss_r, adv_loss_a, nn.ReLU()(self.lambda_recon) * recon_loss]))
        return gen_loss, img_t_hat

    def get_layer_attr_names(self) -> List[str]:
        return ['gen', ]


if __name__ == '__main__':
    _w_in = 64
    _gen = PixelDTGanGenerator(c_in=3, c_out=3, c_hidden=128, n_contracting_blocks=5, c_bottleneck=100, w_in=_w_in,
                               use_dropout=True, use_out_tanh=True)
    get_total_params(_gen, print_table=True, sort_desc=True)
    # print(_gen)
    enable_verbose(_gen)
    _x = torch.randn(1, 3, _w_in, _w_in)
    _y = _gen(_x)

    # State Dict
    print(_gen.state_dict().keys())
    print('lambda_recon' in _gen.state_dict().keys())

    _state_dict = _gen.state_dict()
    del _state_dict['lambda_recon']
    _gen.load_state_dict(_state_dict)
    print(_gen.lambda_recon)
