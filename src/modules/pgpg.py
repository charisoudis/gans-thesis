from typing import Optional, Tuple, Sequence

import numpy as np
import torch
from PIL.Image import Image
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.deep_fashion import ICRBDataset, ICRBCrossPoseDataset
from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.pgpg import PGPGGenerator
from modules.ifaces import IGanGModule
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.plot import create_img_grid, plot_grid
from utils.train import weights_init_naive, get_adam_optimizer, get_optimizer_lr_scheduler


class PGPG(nn.Module, IGanGModule):
    """
    PGPG Class:
    This class is used to access and use the entire PGPG model (implemented according to the paper "Pose-Guided Person
    Image Generation" as a `nn.Module` instance but with the additional functionality provided from inheriting
    `utils.gdrive.GDriveModel` (through the `utils.ifaces.IGanGModule` interface). Inheriting GDriveModel enables easy
    download / upload of model checkpoints to Google Drive using GoogleDrive API's python client and PyDrive.
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
        'gen_opt': {
            'lr': 1e-4,
            'opt': 'adam',
            'scheduler': None
        },
        'disc': {
            'c_hidden': 8,
            'n_contracting_blocks': 5,
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
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, -1),
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
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
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param (Sequence) reproducible_indices: dataset indices to be fetched and visualized each time
                                                `PixelDTGan::visualize(reproducible=True)` is called
        :param (optional) evaluator: GanEvaluator instance of None to not evaluate models when taking snapshots
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        # Initialize interface
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, reproducible_indices=reproducible_indices,
                             evaluator=evaluator, **evaluator_kwargs)

        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define PGPG model
        # This setup leads to 237M (G1 has ~ 120M, G2 has ~117M) learnable parameters for the entire Generator network
        shapes_conf = self._configuration['shapes']
        self.gen = PGPGGenerator(c_in=2 * shapes_conf['c_in'], c_out=shapes_conf['c_out'], w_in=shapes_conf['w_in'],
                                 h_in=shapes_conf['h_in'], configuration=self._configuration['gen'])
        # This setup leads to 396K learnable parameters for the Discriminator network
        # NOTE: for 5 contracting blocks, output is 4x4
        disc_conf = self._configuration['disc']
        self.disc = PatchGANDiscriminator(c_in=2 * shapes_conf['c_in'],
                                          n_contracting_blocks=disc_conf['n_contracting_blocks'],
                                          use_spectral_norm=bool(disc_conf['use_spectral_norm']))
        self.disc_adv_criterion = getattr(nn, f'{disc_conf["adv_criterion"]}Loss')()
        # Move models to GPU
        self.gen.to(device)
        self.disc.to(device)
        self.device = device
        self.is_master_device = (isinstance(device, torch.device) and device.type == 'cuda' and device.index == 0) \
                                or (isinstance(device, torch.device) and device.type == 'cpu') \
                                or (isinstance(device, str) and device == 'cpu')

        # Define optimizers
        # Note: Both generators, G1 & G2, are trained using a joint optimizer
        gen_opt_conf = self._configuration['gen_opt']
        disc_opt_conf = self._configuration['disc_opt']
        self.gen_opt = get_adam_optimizer(self.gen, lr=gen_opt_conf['lr'])
        self.disc_opt = get_adam_optimizer(self.disc, lr=disc_opt_conf['lr'])

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
            self.disc = self.disc.apply(weights_init_naive)
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
                self.disc_opt_lr_scheduler = get_optimizer_lr_scheduler(
                    self.disc_opt, schedule_type=str(disc_opt_conf['scheduler']), base_lr=0.1 * disc_opt_conf['lr'],
                    max_lr=disc_opt_conf['lr'], step_size_up=2 * len(_dataset) if evaluator else 1000, mode='exp_range',
                    gamma=0.9, cycle_momentum=False,
                    last_epoch=chkpt_epoch if 'initial_lr' in self.disc_opt.param_groups[0].keys() else -1)
            else:
                self.disc_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_opt,
                                                                        schedule_type=str(disc_opt_conf['scheduler']))
        else:
            self.disc_opt_lr_scheduler = None

        # Save transforms for visualizer
        if gen_transforms is not None:
            self.gen_transforms = gen_transforms

        # Initialize params
        self.g1_out = None
        self.g_out = None
        self.image_1 = None
        self.image_2 = None
        self.pose_2 = None

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
        self.disc.load_state_dict(state_dict['disc'])
        self.disc_opt.load_state_dict(state_dict['disc_opt'])
        self._nparams = state_dict['nparams']
        # Update latest metrics with checkpoint's metrics
        if 'metrics' in state_dict.keys():
            self.latest_metrics = state_dict['metrics']
        self.logger.debug(f'State dict loaded. Keys: {tuple(state_dict.keys())}')
        for _k in [_k for _k in state_dict.keys() if _k not in ('gen', 'gen_opt', 'disc', 'disc_opt', 'configuration')]:
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

    def forward(self, image_1: Tensor, image_2: Tensor, pose_2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param (Tensor) image_1: the batch of input images as a `torch.Tensor` object
        :param (Tensor) image_2: the batch of target images as a `torch.Tensor` object
        :param (Tensor) pose_2: the batch of target pose images as a `torch.Tensor` object
        :return: a tuple of `torch.Tensor` objects containing (disc_loss, gen_loss)

        -----------
        FROM COLAB:
        ----------

        # Perform forward pass from generator and discriminator
        disc_loss, gen_loss, g1_out, g_out = pgpg(image_1, image_2, pose_2)


        """
        # Update gdrive model state
        if self.is_master_device:
            self.gforward(image_1.shape[0])

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        with self.gen.frozen():
            self.disc_opt.zero_grad()  # Zero out discriminator gradient (before backprop)
            _, g_out = self.gen(image_1, pose_2)
            disc_loss = self.disc.get_loss(real=image_2, fake=g_out.detach(), condition=image_1,
                                           criterion=self.disc_adv_criterion)
            disc_loss.backward(retain_graph=True)  # Update discriminator gradients
            self.disc_opt.step()  # Update discriminator weights
            # Update LR (if needed)
            if self.disc_opt_lr_scheduler:
                if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                    self.disc_opt_lr_scheduler.step(metrics=disc_loss)
                else:
                    self.disc_opt_lr_scheduler.step()

        ##########################################
        ########     Update Generator     ########
        ##########################################
        with self.disc.frozen():
            self.gen_opt.zero_grad()
            g1_loss, g2_loss, g1_out, g_out = self.gen.get_loss(x=image_1, y=image_2, y_pose=pose_2.clone(),
                                                                disc=self.disc, adv_criterion=None,
                                                                recon_criterion=None)
            gen_loss = g1_loss + g2_loss
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
            self.g1_out = g1_out[::len(g1_out) - 1].detach().cpu()
            self.g_out = g_out[::len(g_out) - 1].detach().cpu()
            self.image_1 = image_1[::len(image_1) - 1].detach().cpu()
            self.image_2 = image_2[::len(image_2) - 1].detach().cpu()
            self.pose_2 = pose_2[::len(pose_2) - 1].detach().cpu()
            self.gen_losses.append(gen_loss.item())
            self.disc_losses.append(disc_loss.item())

        return disc_loss, gen_loss

    #
    # --------------
    # Visualizable
    # -------------
    #

    def visualize_indices(self, indices: int or Sequence) -> Image:
        # Fetch images
        assert hasattr(self, 'evaluator') and hasattr(self.evaluator, 'dataset'), 'Could find dataset from model'
        images = []
        with self.gen.frozen():
            for index in indices:
                image_1, image_2, pose_2 = self.evaluator.dataset[index]
                g1_out, g_out = self.gen(image_1.unsqueeze(0), pose_2.unsqueeze(0))
                g1_out = g1_out.squeeze(0)
                g_out = g_out.squeeze(0)
                images.extend([image_1, image_2, self.gen_transforms(pose_2), g1_out, g_out])

        # Convert to grid of images
        ncols = 5
        nrows = int(len(images) / ncols)
        grid = create_img_grid(images=torch.stack(images), nrows=nrows, ncols=ncols, gen_transforms=self.gen_transforms)

        # Plot
        return plot_grid(grid=grid, figsize=(ncols, nrows),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)} | '
                                    f'indices={indices}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses), 3))}, '
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses), 3))}')

    def visualize(self, reproducible: bool = False) -> Image:
        if reproducible:
            return self.visualize_indices(indices=self.reproducible_indices)

        # Get first & last sample from saved images in self
        image_1_0 = self.image_1[0]
        image_2_0 = self.image_2[0]
        pose_2_0 = self.pose_2[0]
        g1_out_0 = self.g1_out[0]
        g_out_0 = self.g_out[0]
        image_1__1 = self.image_1[-1]
        image_2__1 = self.image_2[-1]
        pose_2__1 = self.pose_2[-1]
        g1_out__1 = self.g1_out[-1]
        g_out__1 = self.g_out[-1]

        # Concat images to a 2x5 grid (each row is a separate generation, the columns contain real and generated images
        # side-by-side)
        ncols = 5
        nrows = 2
        grid = create_img_grid(images=torch.stack([
            image_1_0, image_2_0, self.gen_transforms(pose_2_0), g1_out_0, g_out_0,
            image_1__1, image_2__1, self.gen_transforms(pose_2__1), g1_out__1, g_out__1,
        ]), nrows=nrows, ncols=ncols, gen_transforms=self.gen_transforms)

        # Plot
        return plot_grid(grid=grid, figsize=(ncols, nrows),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses), 3))}, '
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses), 3))}')


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
    _gen_transforms = ICRBDataset.get_image_transforms(target_shape=128, target_channels=3)
    _dataset = ICRBCrossPoseDataset(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                                    pose=True, log_level=_log_level)
    _bs = 2
    _dl = DataLoader(dataset=_dataset, batch_size=_bs)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_groot, gen_dataset=_dataset, target_index=1,
                              condition_indices=(0, 2), n_samples=2, batch_size=1,
                              f1_k=1)

    # Initialize model
    _pgpg = PGPG(model_fs_folder_or_root=_models_groot, config_id='128_MSE_256_6_4_5_none_none_1e4_true_false_false',
                 dataset_len=len(_dataset), chkpt_epoch='latest', log_level=_log_level,
                 evaluator=_evaluator, device='cpu')

    # # for _e in range(3):
    # #     _pgpg.logger.info(f'current epoch: {_e}')
    # #     for _i_1, _i_2, _p_2, in iter(_dl):
    # #         _pgpg.gforward(_i_1.shape[0])
    # #
    # # print(json.dumps(_pgpg.list_configurations(only_keys=('title',)), indent=4))

    _device = _pgpg.device
    _x, _y, _y_pose = next(iter(_dl))
    _disc_loss, _gen_loss = _pgpg(_x.to(_device), _y.to(_device), _y_pose.to(_device))
    # print(_disc_loss.shape, _gen_loss.shape, _g1_out.shape, _g_out.shape)

    _img = _pgpg.visualize(reproducible=True)
    _img.show()
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

    _imgs = _pgpg.visualize_metrics(upload=False, preview=True)
    print(_imgs)
