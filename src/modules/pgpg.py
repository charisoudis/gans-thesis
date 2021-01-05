import json
from typing import Optional, Tuple

import torch
import yaml
from torch import nn, Tensor

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.pgpg import PGPGGenerator
from utils.gdrive import GDriveModel, GDriveFolder, GDriveCapsule, GDriveFilesystem
from utils.ifaces import Configurable
from utils.pytorch import get_total_params
from utils.string import to_human_readable
from utils.train import weights_init_naive, get_adam_optimizer


class PGPG(nn.Module, GDriveModel, Configurable):
    """
    PGPG Class:
    This class is used to access and use the entire PGPG model (implemented according to the paper "Pose-Guided Person
    Image Generation" as a `nn.Module` instance but with the additional functionality provided from inheriting
    `utils.gdrive.GDriveModel`. Inheriting GDriveModel we can easily download / upload model checkpoints to
    Google Drive using GoogleDrive API's python client.
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'shapes': {
            'c_in': 3,
            'c_out': 3,
            'w_in': 128,
            'h_in': 128,
        },
        'gen': {
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
        },
        'disc': {
            'c_hidden': 8,
            'n_contracting_blocks': 5,
            'use_spectral_norm': True,
            'adv_criterion': 'MSE',
        },
        'opt': {
            'lr': 1e-4,
            'schedule': None
        }
    }

    def __init__(self, model_gfolder_or_groot: GDriveFolder, config_id: Optional[str] = None,
                 chkpt_step: Optional[int or str] = None, device: str = 'cpu'):
        """
        PGPG class constructor.
        :param (GDriveFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                      checkpoints and metrics from/to Google Drive
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will be
                                         loaded via `nn.Module().load_state_dict()`
        """
        # Instantiate GDriveModel class
        model_name = self.__class__.__name__.lower()
        model_gfolder = model_gfolder_or_groot if model_gfolder_or_groot.name.endswith(model_name) else \
            model_gfolder_or_groot.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=True)
        GDriveModel.__init__(self, model_gfolder=model_gfolder, model_name=model_name)
        # Instantiate InceptionV3 model
        nn.Module.__init__(self)
        # Load model configuration from Google Drive or use default
        if config_id:
            config_filepath = self.fetch_configuration(config_id=config_id)
            with open(config_filepath) as yaml_fp:
                configuration = yaml.load(yaml_fp, Loader=yaml.FullLoader)
            self.config_id = config_id
        else:
            configuration = self.DefaultConfiguration
            self.config_id = None

        # Define PGPG model
        # This setup leads to 237M (G1 has ~ 120M, G2 has ~117M) learnable parameters for the entire Generator network
        shapes_conf = configuration['shapes']
        self.gen = PGPGGenerator(c_in=2 * shapes_conf['c_in'], c_out=shapes_conf['c_out'], w_in=shapes_conf['w_in'],
                                 h_in=shapes_conf['h_in'], configuration=configuration['gen'])
        # This setup leads to 396K learnable parameters for the Discriminator network
        # NOTE: for 5 contracting blocks, output is 4x4
        disc_conf = configuration['disc']
        self.disc = PatchGANDiscriminator(c_in=2 * shapes_conf['c_in'],
                                          n_contracting_blocks=disc_conf['n_contracting_blocks'],
                                          use_spectral_norm=bool(disc_conf['use_spectral_norm']))
        self.disc_adv_criterion = getattr(nn, f'{disc_conf["adv_criterion"]}Loss')()
        # Move models to GPU
        self.gen.to(device)
        self.disc.to(device)
        self.device = device
        # Define optimizers
        # Note: Both generators, G1 & G2, are trained using a joint optimizer
        opt_conf = configuration['opt']
        self.gen_opt = get_adam_optimizer(self.gen, lr=opt_conf['lr'], betas=(0.9, 0.999))
        self.disc_opt = get_adam_optimizer(self.disc, lr=opt_conf['lr'], betas=(0.9, 0.999))
        # Define LR schedulers
        # TODO

        # Load checkpoint from Google Drive
        if chkpt_step:
            chkpt_filepath = self.fetch_checkpoint(step=chkpt_step)
            self.load_state_dict(torch.load(chkpt_filepath, map_location='cpu'))
        else:
            # Initialize weights with small values
            self.gen = self.gen.apply(weights_init_naive)
            self.disc = self.disc.apply(weights_init_naive)
        # Save configuration in instance
        self._configuration = configuration
        self._nparams = None

    def state_dict(self, *args, **kwargs) -> dict:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        return {
            'gen': self.gen.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc': self.disc.state_dict(),
            'disc_opt': self.disc_opt.state_dict(),
            'nparams': self.nparams_hr
        }

    @property
    def nparams(self) -> int or str:
        """
        Get the total numbers of parameters of this model.
        :return: an `int` object
        """
        if not self._nparams:
            self._nparams = get_total_params(self)
        return self._nparams

    @property
    def nparams_hr(self) -> str:
        """
        Get the total numbers of parameters of this model as a human-readable string.
        :return: a `str` object
        """
        return to_human_readable(self.nparams)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_opt.load_state_dict(state_dict['gen_opt'])
        self.disc.load_state_dict(state_dict['disc'])
        self.disc_opt.load_state_dict(state_dict['disc_opt'])

    def configuration(self) -> dict:
        return {**self._configuration, 'nparams': self.nparams, 'nparams_hr': self.nparams_hr}

    def load_configuration(self, configuration: dict) -> None:
        raise NotImplementedError

    def forward(self, image_1: Tensor, image_2: Tensor, pose_2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param (Tensor) image_1: the batch of input images as a `torch.Tensor` object
        :param (Tensor) image_2: the batch of target images as a `torch.Tensor` object
        :param (Tensor) pose_2: the batch of target pose images as a `torch.Tensor` object
        :return: a 4-typle of `torch.Tensor` objects containing (disc_loss, gen_loss, g1_out, g_out)

        -----------
        FROM COLAB:
        ----------

        # Perform forward pass from generator adn discriminator
        disc_loss, gen_loss, g1_out, g_out = pgpg(image_1, image_2, pose_2)


        """
        # Update gdrive model state
        self.gforward(image_1.shape[0])

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        self.disc_opt.zero_grad()  # Zero out gradient before backprop
        with torch.no_grad():
            _, g_out = self.gen(image_1, pose_2)
        disc_loss = self.disc.get_loss(real=image_2, fake=g_out, condition=image_1, criterion=self.disc_adv_criterion)
        disc_loss.backward(retain_graph=True)  # Update discriminator gradients
        self.disc_opt.step()  # Update discriminator weights
        # Update LR (if needed)
        # if disc_lr_scheduler_type and disc_opt_lr_scheduler:
        #     disc_opt_lr_scheduler.step(metrics=disc_loss) if disc_lr_scheduler_type == 'on_plateau' \
        #         else disc_opt_lr_scheduler.step()

        ##########################################
        ########     Update Generator     ########
        ##########################################
        self.gen_opt.zero_grad()
        g1_loss, g2_loss, g1_out, g_out = self.gen.get_loss(x=image_1, y=image_2, y_pose=pose_2.clone(), disc=self.disc,
                                                            adv_criterion=None, recon_criterion=None)
        gen_loss = g1_loss + g2_loss
        gen_loss.backward()  # Update generator gradients

        self.gen_opt.step()  # Update generator optimizer
        # Update LR (if needed)
        # if gen_lr_scheduler_type and gen_opt_lr_scheduler:
        #     gen_opt_lr_scheduler.step(metrics=gen_loss) if gen_lr_scheduler_type == 'on_plateau' \
        #         else gen_opt_lr_scheduler.step()

        return disc_loss, gen_loss, g1_out, g_out


if __name__ == '__main__':
    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _gcapsule = GDriveCapsule(local_gdrive_root=_local_gdrive_root, use_http_cache=True, update_credentials=True)
    _fs = GDriveFilesystem(gcapsule=_gcapsule)
    _groot = GDriveFolder.root(capsule_or_fs=_fs, update_cache=False)

    # Initialize model
    _pgpg = PGPG(model_gfolder_or_groot=_groot, config_id='light', device='cuda')
    print(_pgpg.nparams_hr)
    print(json.dumps(_pgpg.list_all_configurations(only_keys=('title',)), indent=4))

    device = _pgpg.device
    _x = torch.randn(1, 3, 128, 128).to(device)
    _y_pose = torch.randn(1, 3, 128, 128).to(device)
    _, y = _pgpg(_x, _y_pose)
    print(y.shape)

    print(json.dumps(_pgpg.gcapture(checkpoint=True, configuration=True, in_parallel=False), indent=4))
