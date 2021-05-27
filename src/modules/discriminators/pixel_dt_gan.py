from typing import Optional

import torch

from modules.discriminators.patch_gan import PatchGANDiscriminator
from utils.command_line_logger import CommandLineLogger
from utils.pytorch import enable_verbose, get_total_params


class PixelDTGanDiscriminator(PatchGANDiscriminator):
    """
    PixelDTGanDiscriminator Class:
    PixelDTGAN uses DCGAN discriminator but in our implementation we will experiment with PatchGAN discriminator varying
    the patch size (controlled by the number of contracting blocks). Via this way, we can have a PatchGAN up to a
    PixelGAN discriminator and decide on the optimal patch size from the GAN evaluation.
    Parameters:
        c_in: the number of image input channels
        c_hidden: the initial number of discriminator convolutional filters
    """

    def __init__(self, c_in: int, c_hidden: int = 128, n_contracting_blocks: int = 4, use_spectral_norm: bool = False,
                 logger: Optional[CommandLineLogger] = None, adv_criterion: Optional[str] = None):
        """
        PixelDTGanDiscriminator class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_hidden: number of hidden channels
        :param (int) n_contracting_blocks: number of contracting blocks (defaults to 4 as presented in original paper)
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        :param (optional) adv_criterion: str description of desired default adversarial criterion (e.g. 'MSE', 'BCE',
                                         'BCEWithLogits', etc.). If None, then it must be set in the respective function
                                         call.
        """
        super(PixelDTGanDiscriminator, self).__init__(c_in, c_hidden, n_contracting_blocks=n_contracting_blocks,
                                                      use_upfeature=False, use_spectral_norm=use_spectral_norm,
                                                      logger=logger, adv_criterion=adv_criterion)


if __name__ == '__main__':
    # Real/Fake Discriminator
    _disc_r = PixelDTGanDiscriminator(c_in=3, c_hidden=128, n_contracting_blocks=4)
    _disc_r.logger.info('[START] Real/Fake Discriminator')
    # print(_disc)
    enable_verbose(_disc_r)
    _x = torch.randn(1, 3, 64, 64)
    _y = _disc_r(_x)
    _disc_r.logger.info('[END]')

    # Associated/Unassociated Discriminator
    _disc_a = PixelDTGanDiscriminator(c_in=6, c_hidden=128, n_contracting_blocks=4, logger=_disc_r.logger)
    _disc_a.logger.info('[START] Associated/Unassociated Discriminator')
    # print(_disc)
    enable_verbose(_disc_a)
    _x = torch.randn(1, 3, 64, 64)
    _condition = torch.randn(1, 3, 64, 64)
    _ = _disc_a(_x, _condition)
    _disc_a.logger.info('[END]')

    get_total_params(_disc_a, print_table=True, sort_desc=True)
