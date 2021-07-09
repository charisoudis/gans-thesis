from typing import Optional

from modules.discriminators.patch_gan import PatchGANDiscriminator
from utils.command_line_logger import CommandLineLogger


class CycleGANDiscriminator(PatchGANDiscriminator):
    """
    CycleGANDiscriminator Class:
    CycleGAN uses PatchGAN discriminator so this class simply acts as an alias of
    `modules.discriminators.patch_gan.PatchGANDiscriminator`.
    THIS CLASS WAS NEVER USED. INSTEAD USED PatchGANDiscriminator DIRECTLY.
    """

    def __init__(self, c_in: int, c_hidden: int = 8, logger: Optional[CommandLineLogger] = None):
        """
        CycleGANDiscriminator class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_hidden: number of hidden channels
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        super(CycleGANDiscriminator, self).__init__(c_in, c_hidden, n_contracting_blocks=3, logger=logger)
