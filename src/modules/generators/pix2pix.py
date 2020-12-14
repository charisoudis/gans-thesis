from modules.generators.unet import UNETWithSkipConnections


class Pix2PixGenerator(UNETWithSkipConnections):
    """
    Pix2PixGenerator Class:
    The generator NN used in pix2pix is a simple UNETWithSkipConnections with skip connections.
    """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 32, n_contracting_blocks: int = 6,
                 use_dropout: bool = True):
        """
        UNETWithSkipConnections class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param n_contracting_blocks: the base number of contracting (and corresponding expanding) blocks
        :param use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        """
        super(Pix2PixGenerator, self).__init__(c_in=c_in, c_out=c_out, c_hidden=c_hidden,
                                               n_contracting_blocks=n_contracting_blocks,
                                               use_dropout=use_dropout)
