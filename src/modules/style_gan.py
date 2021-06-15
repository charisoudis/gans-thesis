import os
from typing import Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.cuda.amp import GradScaler
from torchvision.transforms import Compose

from datasets.bags2shoes import Bags2ShoesDataloader, Bags2ShoesDataset
from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.ifaces import IGanGModule
from utils.filesystems.local import LocalCapsule, LocalFolder
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.train import get_optimizer, weights_init_naive, set_optimizer_lr


class StyleGAN(nn.Module, IGanGModule):
    """
    StyleGAN Class:
    This is the entire StyleGAN model consisting of one noise-to-image Generator and one PatchGAN Discriminator.
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'shapes': {
            'z_dim': 512,
            'c_in': 3,
            'c_out': 3,
            'w_out': 128,
            'h_out': 128,
        },
        'gen': {
            'c_hidden': 64,
            'n_contracting_blocks': 2,
            'n_residual_blocks': 9,
            'output_padding': 1,
            'use_skip_connections': False,
            'adv_criterion': 'MSE',
            'cycle_criterion': 'L1',
            'identity_criterion': 'L1'
        },
        'gen_opt': {
            'optim_type': 'Adam',
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'scheduler_type': None,
            'betas': (0.5, 0.999)
        },
        'disc': {
            'c_hidden': 64,
            'n_contracting_blocks': 3,
            'use_spectral_norm': True,
            'adv_criterion': 'MSE'
        },
        'disc_opt': {
            'optim_type': 'Adam',
            'lr': 1e-4,
            'weight_decay': 0,
            'scheduler_type': None,
            'betas': (0.5, 0.999),
            'joint_opt': False
        },
        'use_half_precision': True
    }

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = 'default',
                 chkpt_epoch: Optional[int or str] = None, chkpt_step: Optional[int] = None,
                 device: torch.device or str = 'cpu', gen_transforms: Optional[Compose] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, -1),
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
        """
        StyleGAN class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.gdrive.GDriveFolder` object to download /
                                                           upload model checkpoints and metrics from / to local or
                                                           remote (Google Drive) filesystem
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (int or str or None) chkpt_epoch: if not `None` then the model checkpoint at the given :attr:`step` will
                                                 be loaded via `nn.Module().load_state_dict()`
        :param (int or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` and at
                                         the given :attr:`batch_size` will be loaded via `nn.Module().load_state_dict()`
                                         call
        :param (str) device: the device used for training (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (Compose) gen_transforms: the image transforms of the dataset the generator is trained on (used in
                                         visualization)
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param (Sequence) reproducible_indices: dataset indices to be fetched and visualized each time
                                                `PixelDTGan::visualize(reproducible=True)` is called
        :param (optional) evaluator: GanEvaluator instance of None to not evaluate models when taking snapshots
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        # Initialize interface
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id=config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, reproducible_indices=reproducible_indices,
                             evaluator=evaluator, **evaluator_kwargs)

        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define CycleGAN model
        shapes_conf = self._configuration['shapes']
        criteria_conf = {
            'adv': getattr(nn, f'{self._configuration["gen"]["adv_criterion"]}Loss')(),
        }
        #   - Generator
        self.gen = StyleGANGenerator(z_dim=shapes_conf['z_dim'], c_out=shapes_conf['c_out'], logger=self.logger,
                                     criteria_conf=criteria_conf, **self._configuration['gen'])
        #   - Discriminator
        self.disc = PatchGANDiscriminator(c_in=shapes_conf['c_in'], logger=self.logger, **self._configuration['disc'])
        # Move models to {C,G,T}PU
        self.gen.to(device)
        self.disc.to(device)
        self.device = device
        self.is_master_device = (isinstance(device, torch.device) and device.type == 'cuda' and device.index == 0) \
                                or (isinstance(device, torch.device) and device.type == 'cpu') \
                                or (isinstance(device, str) and device == 'cpu')
        #   - Optimizers & LR Schedulers
        self.gen_opt, self.gen_opt_lr_scheduler = get_optimizer(self.gen, **self._configuration['gen_opt'])
        disc_opt_conf = self._configuration['disc_opt']
        self.disc_opt, self.disc_opt_lr_scheduler = get_optimizer(self.disc, **disc_opt_conf)

        # Load checkpoint from Google Drive
        self.other_state_dicts = {}
        if chkpt_epoch is not None:
            try:
                chkpt_filepath = self.fetch_checkpoint(epoch_or_id=chkpt_epoch, step=chkpt_step)
                self.logger.debug(f'Loading checkpoint file: {chkpt_filepath}')
                _state_dict = torch.load(chkpt_filepath, map_location='cpu')
                self.load_state_dict(_state_dict)
                if 'gforward' in _state_dict.keys():
                    self.load_gforward_state(_state_dict['gforward'])
            except FileNotFoundError as e:
                self.logger.critical(str(e))
                chkpt_epoch = None
        if not chkpt_epoch:
            # Initialize weights with small values
            self.gen_a_to_b = self.gen_a_to_b.apply(weights_init_naive)
            self.gen_b_to_a = self.gen_b_to_a.apply(weights_init_naive)
            self.disc_a = self.disc_a.apply(weights_init_naive)
            self.disc_b = self.disc_b.apply(weights_init_naive)
        # For visualizations
        self.fake_a = None
        self.fake_b = None
        self.real_a = None
        self.real_b = None
        self.gen_transforms = gen_transforms

        # Save arguments
        self.device = device
        self.use_half_precision = self._configuration['use_half_precision'] and str(device) != 'cpu' \
            if 'use_half_precision' in self._configuration.keys() else False

        # Create a GradScaler once (for float16 forward/backward passes)
        if self.use_half_precision:
            self.grad_scaler = GradScaler()

    def load_configuration(self, configuration: dict) -> None:
        IGanGModule.load_configuration(self, configuration)

    @property
    def gen(self) -> nn.Module:
        return self.gen_a_to_b

    @gen.setter
    def gen(self, gen: Optional[nn.Module]) -> None:
        pass

    def update_lr(self, gen_new_lr: Optional[float] = None, disc_new_lr: Optional[float] = None) -> None:
        """
        Updates learning-rate of model optimizers, for the non-None give arguments.
        :param (float|None) gen_new_lr: new LR for generator's optimizer (or None to leave as is)
        :param (float|None) disc_new_lr: new LR for real/fake discriminator's optimizer (or None to leave as is)
        """
        if gen_new_lr:
            set_optimizer_lr(self.gen_opt, new_lr=gen_new_lr)
        if disc_new_lr:
            set_optimizer_lr(self.disc_opt, new_lr=disc_new_lr)

    #
    # ------------
    # nn.Module
    # -----------
    #

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        # TODO
        raise NotImplementedError

    def state_dict(self, *args, **kwargs) -> dict:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        mean_gen_loss = np.mean(self.gen_losses)
        self.gen_losses.clear()
        mean_disc_loss = np.mean(self.disc_losses)
        self.disc_losses.clear()
        return {
            'gen': self.gen.state_dict(),
            'gen_loss': mean_gen_loss,
            'gen_opt': self.gen_opt.state_dict(),
            'disc': self.disc.state_dict(),
            'disc_loss': mean_disc_loss,
            'disc_opt': self.disc_opt.state_dict(),
            'nparams': self.nparams,
            'nparams_hr': self.nparams_hr,
            'config_id': self.config_id,
            'configuration': self._configuration,
        }

    def forward(self, real_a: Tensor, real_b: Tensor, lambda_identity: float = 0.1, lambda_cycle: float = 10) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the entire CycleGAN model.
        :param real_a: batch of images from real dataset of domain A
        :param real_b: batch of images from real dataset of domain B
        :param lambda_identity: weight for identity loss in total generator loss
        :param lambda_cycle: weight for cycle-consistency loss in total generator loss
        :return: a tuple containing: 1) joint discriminator loss and 2) joint generators loss
        """
        # Update gdrive model state
        if self.is_master_device:
            self.gforward(real_a.shape[0])

        #############################################
        ########   Update Discriminator(s)   ########
        #############################################
        pass

        #############################################
        ########     Update Generator(s)     ########
        #############################################
        pass

        # TODO
        raise NotImplementedError

    #
    # --------------
    # Visualizable
    # -------------
    #

    def visualize_indices(self, indices: int or Sequence) -> Image:
        # TODO
        raise NotImplementedError

    def visualize(self, reproducible: bool = False) -> Image:
        # TODO
        raise NotImplementedError


if __name__ == '__main__':
    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _log_level = 'debug'

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _groot = LocalFolder.root(capsule_or_fs=LocalCapsule(_local_gdrive_root))

    # Define folder roots
    _models_root = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Initialize model evaluator
    _gen_transforms = Bags2ShoesDataset.get_image_transforms(
        target_shape=StyleGAN.DefaultConfiguration['shapes']['w_in'],
        target_channels=StyleGAN.DefaultConfiguration['shapes']['c_out']
    )

    _bs = 2
    _dl = Bags2ShoesDataloader(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                               log_level=_log_level, batch_size=_bs, count_len_on='min', pin_memory=False)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_root, gen_dataset=_dl.dataset, target_index=1,
                              condition_indices=(0,), n_samples=2, batch_size=1, f1_k=1, device='cpu')

    # Initialize model
    _stgan = StyleGAN(model_fs_folder_or_root=_models_root, config_id='default', chkpt_step=None, chkpt_epoch=None,
                      dataset_len=len(_dl.dataset), log_level=_log_level, evaluator=_evaluator, device='cpu')
    _stgan.use_half_precision = False
    print(_stgan.nparams_hr)
    # exit(0)

    _device = _stgan.device
    _x, _y = next(iter(_dl))
    _disc_loss, _gen_loss = _stgan(_x.to(_device), _y.to(_device))
    print(_disc_loss, _gen_loss)

    _img = _stgan.visualize()
    with open('/home/achariso/Pictures/Thesis/stgan_sample.png', 'wb') as _img_fp:
        _img.save(_img_fp)
        os.system('xdg-open /home/achariso/Pictures/Thesis/stgan_sample.png')
    plt.imshow(_img)
    plt.show()
