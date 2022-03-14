from training.networks_stylegan3 import Generator as NVLabsStyleGan3Generator
from utils.ifaces import BalancedFreezable


class StyleGan3Generator(NVLabsStyleGan3Generator, BalancedFreezable):

    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Init nn.Module
        NVLabsStyleGan3Generator.__init__(self, **locals())
