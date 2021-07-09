from collections import OrderedDict
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import FeatureMapLayer, ChannelsProjectLayer
from modules.partial.encoding import ContractingBlock
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import get_total_params, ReceptiveFieldCalculator
from utils.string import to_human_readable


class PatchGANDiscriminator(nn.Module, BalancedFreezable, Verbosable):
    """
    PatchGANDiscriminator Class:
    This class implements the PatchGAN discriminator network used by many GAN architectures, such as pix2pix, pix2pixHD
    and CycleGAN. Outputs a map of real/fake probabilities instead of scalar per input image.
    """

    def __init__(self, c_in: int, c_hidden: int = 8, n_contracting_blocks: int = 4, use_spectral_norm: bool = False,
                 use_upfeature: bool = True, logger: Optional[CommandLineLogger] = None,
                 adv_criterion: Optional[str] = None):
        """
        PatchGANDiscriminator class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_hidden: the initial number of discriminator convolutional filters (channels)
        :param (int) n_contracting_blocks: number of contracting blocks
        :param (bool) use_spectral_norm: set to True to use Spectral Normalization in the ChannelsProject (last) layer
        :param (bool) use_upfeature: set to True to initially up-channel the input before contracting blocks
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        :param (optional) adv_criterion: str description of desired default adversarial criterion (e.g. 'MSE', 'BCE',
                                         'BCEWithLogits', etc.). If None, then it must be set in the respective function
                                         call.
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        if use_upfeature:
            self.patch_gan_discriminator = nn.Sequential(
                FeatureMapLayer(c_in, c_hidden),

                # Encoding (aka contracting) blocks
                ContractingBlock(c_hidden, use_norm=False),
                *[ContractingBlock(c_hidden * 2 ** i) for i in range(1, n_contracting_blocks)],

                ChannelsProjectLayer(c_hidden * 2 ** n_contracting_blocks, 1, use_spectral_norm=use_spectral_norm)
            )
        else:
            self.patch_gan_discriminator = nn.Sequential(
                # Encoding (aka contracting) blocks
                ContractingBlock(c_in=c_in, c_out=c_hidden, use_norm=False),
                *[ContractingBlock(c_hidden * 2 ** i) for i in range(0, n_contracting_blocks - 1)],

                ChannelsProjectLayer(c_hidden * 2 ** (n_contracting_blocks - 1), 1, use_spectral_norm=use_spectral_norm)
            )

        # Save args
        self.use_upfeature = use_upfeature
        self.n_contracting_blocks = n_contracting_blocks
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger
        self.adv_criterion = getattr(nn, f'{adv_criterion}Loss')() if adv_criterion is not None else None
        self.verbose_enabled = False

    @property
    def nparams_hr(self):
        return to_human_readable(get_total_params(self))

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Function for completing a forward pass of PatchGANDiscriminator:
        Given an image tensor, returns a 2D matrix of realness probabilities for each image's "patches".
        :param (torch.Tensor) x: image tensor of shape (N, C_in, H, W)
        :param (torch.Tensor) y: image tensor of shape (N, C_y, H, W) containing the condition images (e.g. for pix2pix)
        :return: transformed image tensor of shape (N, 1, P_h, P_w)
        """
        if y is not None:
            x = torch.cat([x, y], dim=1)  # channel-wise concatenation
        if self.verbose_enabled:
            self.logger.debug('_: ' + str(x.shape))
        return self.patch_gan_discriminator(x)

    def get_loss(self, real: Tensor, fake: Tensor, condition: Optional[Tensor] = None,
                 criterion: Optional[nn.modules.Module] = None, real_unassoc: Optional[Tensor] = None) -> Tensor:
        """
        Compute adversarial loss.
        :param (torch.Tensor) real: image tensor of shape (N, C, H, W) from real dataset
        :param (torch.Tensor) fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param (optional) condition: condition image tensor of shape (N, C_in/2, H, W) that is stacked before input to
                                     PatchGAN discriminator (optional)
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :param (torch.Tensor) real_unassoc: (to use only for Associated/Unassociated discriminator (e.g. PixelDTGan))
        :return: torch.Tensor containing loss value(s)
        """
        # Setup criterion
        criterion = self.adv_criterion if criterion is None else criterion
        # Proceed with loss calculation
        predictions_on_real = self(real, condition)
        predictions_on_fake = self(fake, condition)
        print(predictions_on_fake.shape)
        # print('DISC OUTPUT SHAPE: ' + str(predictions_on_fake.shape))
        if type(criterion) == torch.nn.modules.loss.BCELoss:
            predictions_on_real = nn.Sigmoid()(predictions_on_real)
            predictions_on_fake = nn.Sigmoid()(predictions_on_fake)
        loss_on_real = criterion(predictions_on_real, torch.ones_like(predictions_on_real))
        loss_on_fake = criterion(predictions_on_fake, torch.zeros_like(predictions_on_fake))
        losses = [loss_on_real, loss_on_fake]
        if real_unassoc is not None:
            predictions_on_real_unassoc = self(real_unassoc[0:condition.shape[0], :, :, :], condition)
            if type(criterion) == torch.nn.modules.loss.BCELoss:
                predictions_on_real_unassoc = nn.Sigmoid()(predictions_on_real_unassoc)
            loss_on_real_unassoc = criterion(predictions_on_real_unassoc, torch.zeros_like(predictions_on_real_unassoc))
            losses.append(loss_on_real_unassoc)
        return torch.mean(torch.stack(losses))

    def get_layer_attr_names(self) -> List[str]:
        return ['patch_gan_discriminator', ]

    def get_receptive_field(self, w_in: int) -> int:
        """
        Returns the receptive field of each of the elements in the output matrix.
        :param (int) w_in: width of input images (ass. that it equals the height)
        :return: an integer object containing the receptive field value
        """
        # Construct architecture dict
        current_arch = []
        if self.use_upfeature:
            current_arch.append(('uf', [7, 1, 3]))
        for bi in range(self.n_contracting_blocks):
            current_arch.append((f'contr_block_{bi}', [3, 2, 1]))
        current_arch.append(('pj', [1, 1, 0]))
        current_arch_dict = OrderedDict(current_arch)
        # Calculate and return
        return min(w_in, ReceptiveFieldCalculator.calculate(current_arch_dict, w_in))


if __name__ == '__main__':
    __disc = PatchGANDiscriminator(c_in=6, n_contracting_blocks=5, use_spectral_norm=True, adv_criterion='BCE')
    _w_in = 128
    
    _real = torch.randn(1, 3, _w_in, _w_in)
    _fake = torch.randn(1, 3, _w_in, _w_in)
    _condition = torch.randn(1, 3, _w_in, _w_in)
    _real_unassoc = torch.randn(1, 3, _w_in, _w_in)
    # print(__disc)
    _loss = __disc.get_loss(real=_real, fake=_fake, condition=_condition, real_unassoc=_real_unassoc)
    print(_loss)
    print(_loss)

    _rf = __disc.get_receptive_field(w_in=_w_in)
    print(f'Receptive Field for input {_w_in}x{_w_in}: {_rf}x{_rf}')
