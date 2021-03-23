from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from datasets.deep_fashion import ICRBCrossPoseDataloader
from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.unet import UNETWithSkipConnections
from utils.filesystems.local import LocalFolder, LocalCapsule
from utils.ifaces import Freezable


class PGPGGenerator1(UNETWithSkipConnections):
    """
    PGPGGenerator1 Class:
    This class implements the G1 generator network from the PGPG paper ("Pose Guided Person Image Generation").
    """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 32, n_contracting_blocks: int = 6,
                 c_bottleneck_down: int = 10, w_in: int = 256, h_in: int = 256, use_out_tanh: bool = True):
        """
        PGPGGenerator1 class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param c_bottleneck_down: (G1) the number of channels to project down to before flattening for FC layer (this is
                                  necessary since otherwise memory will be exhausted), in generator 1 network.
        :param h_in: the input image height
        :param w_in: the input image width
        :param use_out_tanh: set to True to use Tanh() activation in output layer; otherwise no output activation will
                             be used
        """
        super(PGPGGenerator1, self).__init__(c_in=c_in, c_out=c_out, c_hidden=c_hidden, use_dropout=False, use_bn=False,
                                             n_contracting_blocks=n_contracting_blocks, fc_in_bottleneck=True,
                                             h_in=h_in, w_in=w_in, c_bottleneck_down=c_bottleneck_down,
                                             use_out_tanh=use_out_tanh)


class PGPGGenerator2(UNETWithSkipConnections):
    """
    PGPGGenerator2 Class:
    This class implements the G2 generator network from the PGPG paper ("Pose Guided Person Image Generation").
    """

    def __init__(self, c_in: int, c_out: int, c_hidden: int = 32, n_contracting_blocks: int = 5,
                 use_dropout: bool = True, use_out_tanh: bool = True):
        """
        PGPGGenerator2 class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden: the base number of channels multiples of which are used through-out the UNET network
        :param n_contracting_blocks: the base number of contracting (and corresponding expanding) blocks
        :param use_dropout: set to True to use DropOut in the 1st half of the encoder part of the network
        :param use_out_tanh: set to True to use Tanh() activation in output layer; otherwise no output activation will
                             be used
        """
        super(PGPGGenerator2, self).__init__(c_in=c_in, c_out=c_out, c_hidden=c_hidden, use_bn=False,
                                             use_dropout=use_dropout, n_contracting_blocks=n_contracting_blocks,
                                             use_out_tanh=use_out_tanh)


class PGPGGenerator(nn.Module, Freezable):
    """
    PGPGGenerator Class:
    This class implements the whole (2-stage) PGPG generator network similar to the one found in the PGPG paper ("Pose
    Guided Person Image Generation").
    """

    DefaultConfiguration = {
        'g1': {
            'c_hidden': 32,
            'n_contracting_blocks': 6,
            'c_bottleneck_down': 256,
            'use_out_tanh': True,
        },
        'g2': {
            'c_hidden': 32,
            'n_contracting_blocks': 5,
            'use_out_tanh': True,
            'use_dropout': True,
        },
        'recon_criterion': 'L1',
        'adv_criterion': 'MSE',
    }

    def __init__(self, c_in: int, c_out: int, w_in: int, h_in: int, configuration: Optional[dict] = None):
        """
        PGPGGenerator class constructor:
        :param c_in: the number of channels to expect from a given input (image's channels + pose maps' channels)
        :param c_out: the number of channels to expect for a given output
        :param w_in: input image's width
        :param h_in: input image's height
        :param (optional) configuration: the configuration parameters of the networks. Among other parameters the
                                         following are necessary to instantiate the model:
                                            - g1_c_bottleneck_down: number of channels to down-project to on G1's
                                            bottleneck before applying FC layer. Down-projecting to P channels is
                                            necessary to reduce number of params from 1024*Hp*Wp to P*Hp*Wp+1024.
                                            Default is 256.
                                            - use_dropout_in_g2: set to True to apply Dropout in G2's encoder's first
                                            half
        """
        super(PGPGGenerator, self).__init__()

        # Check if configuration is given
        if not configuration:
            configuration = self.DefaultConfiguration

        g1_conf = configuration['g1']
        self.g1 = PGPGGenerator1(c_in=c_in, c_out=c_out, c_hidden=g1_conf['c_hidden'],
                                 n_contracting_blocks=g1_conf['n_contracting_blocks'],
                                 c_bottleneck_down=g1_conf['c_bottleneck_down'], w_in=w_in, h_in=h_in,
                                 use_out_tanh=g1_conf['use_out_tanh'])
        g2_conf = configuration['g2']
        self.g2 = PGPGGenerator2(c_in=2 * c_out, c_out=c_out, c_hidden=g2_conf['c_hidden'],
                                 n_contracting_blocks=g2_conf['n_contracting_blocks'],
                                 use_dropout=g2_conf['use_dropout'],
                                 use_out_tanh=g2_conf['use_out_tanh'])
        self.output_activation = nn.Tanh()

        # Save configuration in instance
        self.configuration = configuration

    def forward(self, x: Tensor, y_pose: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Method to perform a forward pass through PGPG generator.
        :param x: the input image batch
        :param y_pose: the condition (i.e. the target pose) image batch
        :return: a tuple containing G1's output and G2's output (i.e. the generated image)
        """
        g1_out = self.g1(torch.cat((x, y_pose), dim=1))
        with self.g1.frozen():
            g2_out = self.g2(torch.cat((x, g1_out), dim=1))
            g2_out = self.output_activation(g2_out + g1_out)
        return g1_out, g2_out

    CHECK_POSE_TICKS = 10

    def get_loss(self, x: Tensor, y_pose: Tensor, y: Tensor, disc: nn.Module,
                 adv_criterion: Optional[nn.modules.Module] = None, lambda_recon: int = 10,
                 recon_criterion: Optional[nn.Module] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get the loss of the generator given inputs. If the criteria are not provided they will be set using the
        instance's given (or default) configuration.
        :param x: input images
        :param y_pose: target images' pose images
        :param y: target images
        :param disc: the Discriminator network
        :param (optional) adv_criterion: the adversarial loss function; takes the discriminator predictions and the
                                         target labels and returns a adversarial loss (which we aim to minimize)
        :param (int) lambda_recon: weighting factor of L1 loss to calculate G2's final loss
        :param (optional) recon_criterion: the reconstruction loss function; takes the real images from Y and those
                                           images put through a (X,Yp)->Y generator and returns the image reconstruction
                                           loss (which we aim to minimize)
        :return: a tuple containing the G1's loss (a scalar) and G2's loss (a scalar) and the outputs (for visualization
                 purposes)
        """
        # 1) Make a forward pass on the Generator
        g1_out, g_out = self(x, y_pose)
        #   - check pose encoding
        # if self.CHECK_POSE_TICKS == 0 and False:
        #     shape = y_pose.size()
        #     # TODO: [FIX] RuntimeError: shape '[1, 3, 16384]' is invalid for input of size 2359296
        #     values, _ = y_pose.view((1, shape[1], shape[2] * shape[2])).max(dim=2)  # <-- HERE
        #     assert all([v.item() < 1.0 for v in values.view(-1)])
        #     values, _ = y_pose.reshape((1, shape[1], shape[2] * shape[2])).min(dim=2)
        #     assert all([v.item() == 0.0 for v in values.view(-1)])
        #     self.CHECK_POSE_TICKS = 10
        # else:
        #     self.CHECK_POSE_TICKS -= 1
        #   - preview pose image
        # save_tensor_to_image_file(y_pose)
        #   - reinforce non-background parts based on pose (DensePose has 0 in the background pixels)
        y_pose[y_pose > 0] = 1  # pose acts as a loss mask since it is a DensePose IUV map, not just skeleton points
        y_pose += 2  # how much we want to weight on non-background area (original paper weight is 1)
        # 2) Compute Generator Loss
        #   - L1 loss for G1
        recon_criterion = getattr(nn, f'{self.configuration["recon_criterion"]}Loss')() if not recon_criterion \
            else recon_criterion
        g1_loss = recon_criterion(g1_out * y_pose, y * y_pose)
        #   - L1 loss for G2
        g2_loss_recon = recon_criterion(g_out * y_pose, y * y_pose)
        #   - Adversarial loss for G2
        gen_out_predictions = disc(g_out, x)
        adv_criterion = getattr(nn, f'{self.configuration["adv_criterion"]}Loss')() if not adv_criterion else \
            adv_criterion
        g2_loss_adv = adv_criterion(gen_out_predictions, torch.ones_like(gen_out_predictions))
        # - Aggregated loss
        g2_loss = g2_loss_adv + lambda_recon * g2_loss_recon
        return g1_loss, g2_loss, g1_out, g_out

    def freeze(self) -> None:
        self.g1.freeze()
        self.g2.freeze()

    def unfreeze(self) -> None:
        self.g2.unfreeze()
        self.g1.unfreeze()


if __name__ == '__main__':
    __target_channels = 3
    __gen = PGPGGenerator(c_in=2 * __target_channels, c_out=__target_channels, w_in=128, h_in=128).to('cpu')
    __disc = PatchGANDiscriminator(c_in=2 * __target_channels, n_contracting_blocks=5, use_spectral_norm=True).to('cpu')
    __datasets_folder = LocalFolder.root(
        LocalCapsule('/home/achariso/PycharmProjects/gans-thesis/.gdrive')).subfolder_by_name('Datasets')
    __dl = ICRBCrossPoseDataloader(
        dataset_fs_folder_or_root=__datasets_folder,
        batch_size=1, target_shape=128, target_channels=__target_channels, pin_memory=False)
    for _, __images in enumerate(__dl):
        __x, __y, __y_pose = __images
        print(__x.shape)
        print(__y.shape)
        print(__y_pose.shape)

        __g1_loss, __g_loss, _, _ = __gen.get_loss(__x, __y_pose, __y, disc=__disc)
        print(__g1_loss)
        print(__g_loss)

        # Visualization Code
        # from torchviz import make_dot, make_dot_from_trace
        # dot = make_dot(__g2_out, params=dict(__gen.named_parameters()))
        # dot.render("test.png")
        # print(dot)
        break
