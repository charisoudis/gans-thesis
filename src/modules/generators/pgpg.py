from torch import nn, Tensor

from modules.generators.unet import UNETWithSkipConnections


class PGPGGenerator1(UNETWithSkipConnections):
    """
    PGPGGenerator1 Class:
    This class implements the G1 generator network from the PGPG paper ("Pose Guided Person Image Generation").
    """

    def __init__(self, c_in, c_out, c_hidden: int = 32, c_bottleneck_down: int = 10, w_in: int = 256, h_in: int = 256):
        """
        PGPGGenerator1 class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param c_bottleneck_down: (G1) the number of channels to project down to before flattening for FC layer (this is
                                  necessary since otherwise memory will be exhausted), in generator 1 network.
        :param h_in: the input image height
        :param w_in: the input image width
        """
        super(PGPGGenerator1, self).__init__(c_in=c_in, c_out=c_out, c_hidden=c_hidden, n_contracting_blocks=4,
                                             use_dropout=False, fc_in_bottleneck=True, h_in=h_in, w_in=w_in,
                                             c_bottleneck_down=c_bottleneck_down)


class PGPGGenerator2(UNETWithSkipConnections):
    """
    PGPGGenerator2 Class:
    This class implements the G2 generator network from the PGPG paper ("Pose Guided Person Image Generation").
    """

    def __init__(self, c_in, c_out, c_hidden: int = 32, n_contracting_blocks: int = 6, use_dropout: bool = True):
        """
        PGPGGenerator2 class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param n_contracting_blocks: the base number of contracting (and corresponding expanding) blocks
        :param use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        """
        super(PGPGGenerator2, self).__init__(c_in=c_in, c_out=c_out, c_hidden=c_hidden,
                                             n_contracting_blocks=n_contracting_blocks,
                                             use_dropout=use_dropout)


class PGPGGenerator(nn.Module):
    # TODO: implement whole (2-stage) PGPG generator module

    def __init__(self):
        super(PGPGGenerator, self).__init__()

    def forward(self, x: Tensor, y_pose: Tensor) -> Tensor:
        pass

    def get_loss(self):
        pass
