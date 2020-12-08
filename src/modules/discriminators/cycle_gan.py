from modules.discriminators.patch_gan import PatchGANDiscriminator


class CycleGANDiscriminator(PatchGANDiscriminator):
    """
    CycleGANDiscriminator Class:
    CycleGAN uses PatchGAN discriminator so this class simply acts as an alias
    Parameters:
        c_in: the number of image input channels
        c_hidden: the initial number of discriminator convolutional filters
    """

    def __init__(self, c_in: int, c_hidden: int = 8):
        """
        CycleGANDiscriminator class constructor.
        :param c_in: number of input channels
        :param c_hidden: number of hidden channels
        """
        super(CycleGANDiscriminator, self).__init__(c_in, c_hidden, n_contracting_blocks=3)
