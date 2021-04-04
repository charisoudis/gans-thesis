from typing import Optional, Tuple

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torch.nn import L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.look_book import PixelDTDataset
from modules.discriminators.pixel_dt_gan import PixelDTGanDiscriminator
from modules.generators.pixel_dt_gan import PixelDTGanGenerator
from modules.ifaces import IGanGModule
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.plot import pltfig_to_pil
from utils.pytorch import invert_transforms
from utils.train import weights_init_naive, get_adam_optimizer, get_optimizer_lr_scheduler


class PixelDTGan(nn.Module, IGanGModule):
    """
    PixelDTGan Class:
    This class is used to access and use the entire PixelDTGan model (implemented according to the paper "Pixel-Level
    Domain Transfer") as a `nn.Module` instance but with the additional functionality provided from inheriting
    `utils.gdrive.GDriveModel` (through the `utils.ifaces.IGanGModule` interface). Inheriting GDriveModel enables easy
    download / upload of model checkpoints to Google Drive using GoogleDrive API's python client and PyDrive
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'shapes': {
            'c_in': 3,
            'c_out': 3,
            'w_in': 64,
        },
        'gen': {
            'c_hidden': 128,
            'n_contracting_blocks': 5,
            'c_bottleneck': 100,
            'use_out_tanh': True,
            'use_dropout': False,
        },
        'gen_opt': {
            'lr': 1e-4,
            'opt': 'adam',
            'scheduler': None
        },
        'disc_r': {
            'c_hidden': 128,
            'n_contracting_blocks': 4,
            'use_spectral_norm': True,
            'adv_criterion': 'MSE',
        },
        'disc_a': {
            'c_hidden': 128,
            'n_contracting_blocks': 4,
            'use_spectral_norm': True,
            'adv_criterion': 'MSE',
        },
        'disc_opt': {
            'lr': 1e-4,
            'opt': 'adam',
            'scheduler': None
        }
    }

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 chkpt_epoch: Optional[int or str] = None, chkpt_step: Optional[int] = None,
                 device: torch.device or str = 'cpu', gen_transforms: Optional[Compose] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
        """
        PGPG class constructor.
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
        :param (optional) evaluator: GanEvaluator instance of None to not evaluate models when taking snapshots
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        # Initialize interface
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, evaluator=evaluator, **evaluator_kwargs)

        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define PixelDTGan model
        # This setup leads to ~38M learnable parameters for the Generator network
        shapes_conf = self._configuration['shapes']
        gen_conf = {**self._configuration['gen'], **{
            'w_in': shapes_conf['w_in'],
            'adv_criterion_conf': {
                'real': getattr(nn, f'{self._configuration["disc_r"]["adv_criterion"]}Loss')(),
                'associated': getattr(nn, f'{self._configuration["disc_a"]["adv_criterion"]}Loss')(),
            }
        }}
        self.gen = PixelDTGanGenerator(c_in=shapes_conf['c_in'], c_out=shapes_conf['c_out'], **gen_conf)
        # get_total_params(self.gen, print_table=True, sort_desc=True)

        # This setup leads to ~6M learnable parameters for the Real/Fake Discriminator network
        # NOTE: for 5 contracting blocks, output is 4x4
        disc_r_conf = self._configuration['disc_r']
        del disc_r_conf['adv_criterion']
        self.disc_r = PixelDTGanDiscriminator(c_in=shapes_conf['c_in'], logger=self.logger, **disc_r_conf)
        self.disc_r_adv_criterion = gen_conf['adv_criterion_conf']['real']

        # This setup leads to ~6M learnable parameters for the Associated/Unassociated Discriminator network
        # NOTE: for 5 contracting blocks, output is 4x4
        disc_a_conf = self._configuration['disc_a']
        del disc_a_conf['adv_criterion']
        self.disc_a = PixelDTGanDiscriminator(c_in=2 * shapes_conf['c_in'], logger=self.logger, **disc_a_conf)
        self.disc_a_adv_criterion = gen_conf['adv_criterion_conf']['associated']
        # get_total_params(self.disc_a, print_table=True, sort_desc=True)

        # Move models to GPU
        self.gen.to(device)
        self.disc_r.to(device)
        self.disc_a.to(device)
        self.device = device
        self.is_master_device = (isinstance(device, torch.device) and device.type == 'cuda' and device.index == 0) \
                                or (isinstance(device, torch.device) and device.type == 'cpu') \
                                or (isinstance(device, str) and device == 'cpu')

        # Define optimizers
        # Note: Both generators, G1 & G2, are trained using a joint optimizer
        gen_opt_conf = self._configuration['gen_opt']
        disc_opt_conf = self._configuration['disc_opt']
        self.gen_opt = get_adam_optimizer(self.gen, lr=gen_opt_conf['lr'])
        self.disc_r_opt = get_adam_optimizer(self.disc_r, lr=disc_opt_conf['lr'])
        self.disc_a_opt = get_adam_optimizer(self.disc_a, lr=disc_opt_conf['lr'])

        # Load checkpoint from Google Drive
        self.other_state_dicts = {}
        if chkpt_epoch:
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
            self.gen = self.gen.apply(weights_init_naive)
            self.disc_r = self.disc_r.apply(weights_init_naive)
            self.disc_a = self.disc_a.apply(weights_init_naive)
            chkpt_epoch = 0

        # Define LR schedulers (after optimizer checkpoints have been loaded)
        if gen_opt_conf['scheduler']:
            if gen_opt_conf['scheduler'] == 'cyclic':
                self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(
                    self.gen_opt, schedule_type=str(gen_opt_conf['scheduler']), base_lr=0.1 * gen_opt_conf['lr'],
                    max_lr=gen_opt_conf['lr'], step_size_up=2 * dataset_len if evaluator else 1000, mode='exp_range',
                    gamma=0.9, cycle_momentum=False,
                    last_epoch=chkpt_epoch if 'initial_lr' in self.gen_opt.param_groups[0].keys() else -1)
            else:
                self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(self.gen_opt,
                                                                       schedule_type=str(gen_opt_conf['scheduler']))
        else:
            self.gen_opt_lr_scheduler = None
        if disc_opt_conf['scheduler']:
            if disc_opt_conf['scheduler'] == 'cyclic':
                self.disc_r_opt_lr_scheduler = get_optimizer_lr_scheduler(
                    self.disc_r_opt, schedule_type=str(disc_opt_conf['scheduler']), base_lr=0.1 * disc_opt_conf['lr'],
                    max_lr=disc_opt_conf['lr'], step_size_up=2 * dataset_len if evaluator else 1000, mode='exp_range',
                    gamma=0.9, cycle_momentum=False,
                    last_epoch=chkpt_epoch if 'initial_lr' in self.disc_r_opt.param_groups[0].keys() else -1)
                self.disc_a_opt_lr_scheduler = get_optimizer_lr_scheduler(
                    self.disc_a_opt, schedule_type=str(disc_opt_conf['scheduler']), base_lr=0.1 * disc_opt_conf['lr'],
                    max_lr=disc_opt_conf['lr'], step_size_up=2 * dataset_len if evaluator else 1000, mode='exp_range',
                    gamma=0.9, cycle_momentum=False,
                    last_epoch=chkpt_epoch if 'initial_lr' in self.disc_a_opt.param_groups[0].keys() else -1)
            else:
                self.disc_r_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_r_opt,
                                                                          schedule_type=str(disc_opt_conf['scheduler']))
                self.disc_a_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_a_opt,
                                                                          schedule_type=str(disc_opt_conf['scheduler']))
        else:
            self.disc_r_opt_lr_scheduler = None
            self.disc_a_opt_lr_scheduler = None

        # Save transforms for visualizer
        if gen_transforms is not None:
            self.gen_transforms = gen_transforms

        # Initialize params
        self.img_s = None
        self.img_t = None
        self.img_t_hat = None
        self.disc_r_losses = []
        self.disc_a_losses = []
        self.img_t_prev = None

    def load_configuration(self, configuration: dict) -> None:
        IGanGModule.load_configuration(self, configuration)

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
        # Check if checkpoint is for different config
        if 'config_id' in state_dict.keys() and state_dict['config_id'] != self.config_id:
            self.logger.critical(f'Config IDs mismatch (self: "{self.config_id}", state_dict: '
                                 f'"{state_dict["config_id"]}"). NOT applying checkpoint.')
            return
        # Load model checkpoints
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_opt.load_state_dict(state_dict['gen_opt'])
        self.disc_r.load_state_dict(state_dict['disc_r'])
        self.disc_r_opt.load_state_dict(state_dict['disc_r_opt'])
        self.disc_a.load_state_dict(state_dict['disc_a'])
        self.disc_a_opt.load_state_dict(state_dict['disc_a_opt'])
        self._nparams = state_dict['nparams']
        # Update latest metrics with checkpoint's metrics
        if 'metrics' in state_dict.keys():
            self.latest_metrics = state_dict['metrics']
        self.logger.debug(f'State dict loaded. Keys: {tuple(state_dict.keys())}')
        for _k in [_k for _k in state_dict.keys() if _k not in ('gen', 'gen_opt', 'disc_r', 'disc_r_opt', 'disc_a',
                                                                'disc_a_opt', 'configuration')]:
            self.other_state_dicts[_k] = state_dict[_k]

    def state_dict(self, *args, **kwargs) -> dict:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        mean_gen_loss = np.mean(self.gen_losses)
        self.gen_losses.clear()
        mean_disc_r_loss = np.mean(self.disc_r_losses)
        self.disc_r_losses.clear()
        mean_disc_a_loss = np.mean(self.disc_a_losses)
        self.disc_a_losses.clear()
        return {
            'gen': self.gen.state_dict(),
            'gen_loss': mean_gen_loss,
            'gen_opt': self.gen_opt.state_dict(),
            'disc_r': self.disc_r.state_dict(),
            'disc_r_loss': mean_disc_r_loss,
            'disc_r_opt': self.disc_r_opt.state_dict(),
            'disc_a': self.disc_a.state_dict(),
            'disc_a_loss': mean_disc_a_loss,
            'disc_a_opt': self.disc_a_opt.state_dict(),
            'nparams': self.nparams,
            'nparams_hr': self.nparams_hr,
            'config_id': self.config_id,
            'configuration': self._configuration,
        }

    def forward(self, img_s: torch.Tensor, img_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param (torch.Tensor) img_s: the batch of input images as a `torch.Tensor` object
        :param (torch.Tensor) img_t: the batch of target images as a `torch.Tensor` object
        :return: a tuple of `torch.Tensor` objects containing (disc_r_loss, disc_a_loss, gen_loss)
        """
        # Update gdrive model state
        if self.is_master_device:
            self.gforward(img_s.shape[0])

        ##########################################
        ########  Update Discriminators   ########
        ##########################################
        with self.gen.frozen():
            disc_loss = {'r': None, 'a': None}
            for disc_i in ['r', 'a']:
                # Get correct discriminator
                disc = getattr(self, f'disc_{disc_i}')
                disc_opt = getattr(self, f'disc_{disc_i}_opt')
                disc_opt_lr_scheduler = getattr(self, f'disc_{disc_i}_opt_lr_scheduler')

                # Perform optimizing step
                disc_opt.zero_grad()  # Zero out discriminator gradient (before backprop)
                img_t_hat = self.gen(img_s)
                disc_loss[disc_i] = disc.get_loss(real=img_t.clone(), fake=img_t_hat.detach(),
                                                  condition=img_s if disc_i == 'a' else None,
                                                  real_unassoc=self.img_t_prev if disc_i == 'a' else None,
                                                  criterion=getattr(self, f'disc_{disc_i}_adv_criterion'))
                disc_loss[disc_i].backward(retain_graph=True)  # Update discriminator gradients
                disc_opt.step()  # Update discriminator weights
                # Update LR (if needed)
                if disc_opt_lr_scheduler:
                    if isinstance(disc_opt_lr_scheduler, ReduceLROnPlateau):
                        disc_opt_lr_scheduler.step(metrics=disc_loss[disc_i])
                    else:
                        disc_opt_lr_scheduler.step()

            # Update img_t_prev
            if self.epoch_inc:
                self.img_t_prev = None
            elif np.random.choice([True, False], 1, p=[0.6, 0.4])[0]:
                self.img_t_prev = img_t.clone()

        ##########################################
        ########     Update Generator     ########
        ##########################################
        with self.disc_r.frozen(), self.disc_a.frozen():
            self.gen_opt.zero_grad()
            gen_loss, img_t_hat = self.gen.get_loss(img_s=img_s, img_t=img_t.clone(), disc_r=self.disc_r,
                                                    disc_a=self.disc_a, adv_criterion=None, recon_criterion=L1Loss())
            gen_loss.backward()  # Update generator gradients
            self.gen_opt.step()  # Update generator optimizer
            # Update LR (if needed)
            if self.gen_opt_lr_scheduler:
                if isinstance(self.gen_opt_lr_scheduler, ReduceLROnPlateau):
                    self.gen_opt_lr_scheduler.step(metrics=gen_loss)
                else:
                    self.gen_opt_lr_scheduler.step()

        # Save for visualization
        if self.is_master_device:
            self.img_s = img_s[::len(img_s) - 1].detach().cpu()
            self.img_t = img_t[::len(img_t) - 1].detach().cpu()
            self.img_t_hat = img_t_hat[::len(img_t_hat) - 1].detach().cpu()
            self.gen_losses.append(gen_loss.item())
            for disc_i in ['r', 'a']:
                disc_losses = getattr(self, f'disc_{disc_i}_losses')
                disc_losses.append(disc_loss[disc_i].item())

        return disc_loss['r'], disc_loss['a'], gen_loss

    #
    # --------------
    # Visualizable
    # -------------
    #

    # noinspection DuplicatedCode
    def visualize(self) -> Image:
        # Inverse generator transforms
        gen_transforms_inv = invert_transforms(self.gen_transforms)
        # Apply inverse image transforms to generated images
        img_t_hat_first = gen_transforms_inv(self.img_t_hat[0]).float()
        img_t_hat_last = gen_transforms_inv(self.img_t_hat[-1]).float()
        # Apply inverse image transforms to real images
        img_s_first = gen_transforms_inv(self.img_s[0]).float()
        img_s_last = gen_transforms_inv(self.img_s[-1]).float()
        img_t_first = gen_transforms_inv(self.img_t[0]).float()
        img_t_last = gen_transforms_inv(self.img_t[-1]).float()
        # Concat images to a 2x5 grid (each row is a separate generation, the columns contain real and generated images
        # side-by-side)
        border = 2
        black = 0.5
        cat_images_1 = torch.cat((
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_s_first,
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_t_first,
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_t_hat_first,
            black * torch.ones(3, img_s_first.shape[1], border).float()
        ), dim=2).cpu()
        cat_images_2 = torch.cat((
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_s_last,
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_t_last,
            black * torch.ones(3, img_s_first.shape[1], border).float(),
            img_t_hat_last,
            black * torch.ones(3, img_s_first.shape[1], border).float()
        ), dim=2).cpu()

        cat_images = torch.cat((
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            cat_images_1,
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            1.0 * torch.ones(3, 4 * border, cat_images_1.shape[2]).float(),
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            cat_images_2,
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
        ), dim=1)

        # Set matplotlib params
        matplotlib.rcParams["font.family"] = 'JetBrains Mono'
        # Create a new figure
        plt.figure(figsize=(5, 2), dpi=300, frameon=False, clear=True)
        # Remove annoying axes
        plt.axis('off')
        # Create image and return
        footer_title = f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}' + ' ' * 76 + \
                       f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses), 3))}, ' \
                       f'disc_loss=(r: {"{0:0.3f}".format(round(np.mean(self.disc_r_losses), 3))}, ' \
                       f'a: {"{0:0.3f}".format(round(np.mean(self.disc_a_losses), 3))})'
        plt.suptitle(footer_title, y=0.03, fontsize=4, fontweight='light', horizontalalignment='left', x=0.001)
        plt.imshow(cat_images.permute(1, 2, 0))
        return pltfig_to_pil(plt.gcf())


if __name__ == '__main__':
    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _log_level = 'debug'

    # # Via GoogleDrive API
    # _groot = GDriveFolder.root(capsule_or_fs=GDriveCapsule(local_gdrive_root=_local_gdrive_root, use_http_cache=True,
    #                                                        update_credentials=True, use_refresh_token=True),
    #                            update_cache=True)
    # ensure_matplotlib_fonts_exist(_groot, force_rebuild=False)

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _fs = LocalFilesystem(LocalCapsule(_local_gdrive_root))
    _groot = LocalFolder.root(capsule_or_fs=_fs)

    # Define folder roots
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Initialize model evaluator
    _gen_transforms = PixelDTDataset.get_image_transforms(
        target_shape=PixelDTGan.DefaultConfiguration['shapes']['w_in'], target_channels=3)
    _dataset = PixelDTDataset(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                              log_level=_log_level)
    _bs = 2
    _dl = DataLoader(dataset=_dataset, batch_size=_bs)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_groot, gen_dataset=_dataset, target_index=1,
                              condition_indices=(0,), n_samples=2, batch_size=1,
                              f1_k=1)

    # Initialize model
    _pxldt = PixelDTGan(model_fs_folder_or_root=_models_groot, config_id='default', dataset_len=len(_dataset),
                        chkpt_epoch=None, log_level=_log_level, evaluator=_evaluator, device='cpu')
    # print(_pxldt)

    # # for _e in range(3):
    # #     _pgpg.logger.info(f'current epoch: {_e}')
    # #     for _i_1, _i_2, _p_2, in iter(_dl):
    # #         _pgpg.gforward(_i_1.shape[0])
    # #
    # # print(json.dumps(_pgpg.list_configurations(only_keys=('title',)), indent=4))

    _device = _pxldt.device
    _x, _y = next(iter(_dl))
    _disc_r_loss, _disc_a_loss, _gen_loss = _pxldt(_x.to(_device), _y.to(_device))
    print(_disc_r_loss, _disc_a_loss, _gen_loss)

    _img = _pxldt.visualize()
    # _img.show()
    with open('sample.png', 'wb') as fp:
        _img.save(fp)
        _pxldt.logger.debug('Image saved.')
    exit(0)

    # import time
    #
    # print('starting capturing...')
    # _async_results = _pgpg.gcapture(checkpoint=True, metrics=True, visualizations=True, configuration=False,
    #                                 in_parallel=True, show_progress=True)
    # for i in range(20):
    #     ready = all(_r.ready() for _r in _async_results)
    #     if not ready:
    #         print('Not ready: sleeping...')
    #         time.sleep(1)
    #     else:
    #         break
    # _uploaded_gfiles = [_r.get() for _r in _async_results]
    # print(json.dumps(_uploaded_gfiles, indent=4))

    _imgs = _pxldt.visualize_metrics(upload=False, preview=True)
    print(_imgs)
