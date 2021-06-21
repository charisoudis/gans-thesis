import gc
import math
import os
import time
from typing import Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from datasets.deep_fashion import FISBDataloader, FISBDataset
from modules.discriminators.stylegan import StyleGanDiscriminator
from modules.generators.stylegan import StyleGanGenerator
from modules.ifaces import IGanGModule
from utils.filesystems.local import LocalCapsule, LocalFolder
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.plot import create_img_grid, plot_grid
from utils.train import get_optimizer, weights_init_naive, set_optimizer_lr


class StyleGan(nn.Module, IGanGModule):
    """
    StyleGan Class:
    This is the entire StyleGan model consisting of one noise-to-image Generator and one PatchGAN Discriminator.
    """

    DefaultConfiguration = {
        'shapes': {
            'c_in': 3,
            'c_out': 3,
        },
        'gen': {
            'z_dim': 512,
            'map_hidden_dim': 512,
            'map_n_blocks': 4,
            'w_dim': 512,
            'c_const': 512,
            'c_hidden': 1024,
            'kernel_size': 3,
            'alpha_multiplier': 10.0,
        },
        'gen_opt': {
            'optim_type': 'Adam',
            'lr': 2e-4,
            'betas': (0.5, 0.999),
            'weight_decay': 0,
            'scheduler_type': 'on_plateau',
            'scheduler_kwargs': {
                'factor': 0.999,  # 1000ppm decrease
                'min_lr': 0.000005,
                'cooldown': 200,
            },
        },
        'disc': {
            'c_base': 2048,
            'c_max': 512,
            'c_decay': 1.0,
            'alpha_multiplier': 10.0,
            'adv_criterion': 'Wasserstein',
            'use_gradient_penalty': True,
            'lambda_gp': 10.0
        },
        'disc_opt': {
            'optim_type': 'Adam',
            'lr': 2e-4,
            'betas': (0.5, 0.999),
            'weight_decay': 0,
            'scheduler_type': 'on_plateau',
            'scheduler_kwargs': {
                'factor': 0.999,  # 1000ppm decrease
                'min_lr': 0.000005,
                'cooldown': 200,
            },
        },
        'resolutions': {
            'min': 4,
            'max': 128,
        },
        'grow_scheduler': {
            'starting_epochs': [
                5,  # at that epoch grow to 8x8
                15,  # at that epoch grow to 16x16
                25,  # at that epoch grow to 32x32
                35,  # at that epoch grow to 64x64
                45,  # at that epoch grow to 128x128
            ],
            'batch_sizes': [
                128,  # images/batch at 4x4
                64,  # images/batch at 8x8
                32,  # images/batch at 16x16
                32,  # images/batch at 32x32
                32,  # images/batch at 64x64
                16,  # images/batch at 128x128
            ],
            'transition_epochs': [
                5,  # epochs for alpha to go from 0 --> 1 at 8x8
                5,  # epochs for alpha to go from 0 --> 1 at 16x16
                5,  # epochs for alpha to go from 0 --> 1 at 32x32
                1,  # epochs for alpha to go from 0 --> 1 at 64x64
                1,  # epochs for alpha to go from 0 --> 1 at 128x128
            ]
        },
        'disc_iters': 1,
        'use_half_precision': True
    }

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = 'default',
                 chkpt_epoch: Optional[int or str] = None, chkpt_step: Optional[int] = None, disc_iters: int = 1,
                 device: torch.device or str = 'cpu', gen_transforms: Optional[Compose] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, 1, -1),
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
        """
        StyleGan class constructor.
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
        :param (int) disc_iters: number of iterations for discriminator (probably only used if Wasserstein Loss is used)
        :param (str) device: the device used for training (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
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

        # Move models to {C,G,T}PU
        self.gen = None
        self.disc = None
        self.device = device
        self.is_master_device = (isinstance(device, torch.device) and device.type == 'cuda' and device.index == 0) \
                                or (isinstance(device, torch.device) and device.type == 'cpu') \
                                or (isinstance(device, str) and device == 'cpu')

        # Load checkpoint from Google Drive
        self.other_state_dicts = {}
        if chkpt_epoch is not None:
            try:
                chkpt_filepath = self.fetch_checkpoint(epoch_or_id=chkpt_epoch, step=chkpt_step)
                self.logger.debug(f'Loading checkpoint file: {chkpt_filepath}')
                state_dict = torch.load(chkpt_filepath, map_location='cpu')
                self._init_gen_disc_opt_scheduler(resolution=state_dict['resolution'], device=device)
                self.load_state_dict(state_dict)
                if 'gforward' in state_dict.keys():
                    self.load_gforward_state(state_dict['gforward'])
            except FileNotFoundError as e:
                self.logger.critical(str(e))
                chkpt_epoch = None
        if not chkpt_epoch:
            # Move to GPU only if no checkpoint is applied
            self._init_gen_disc_opt_scheduler(resolution=self._configuration['resolutions']['min'], device=device)
            # Initialize weights with small values
            if self.gen is not None:
                self.gen = self.gen.apply(weights_init_naive)
            if self.disc is not None:
                self.disc = self.disc.apply(weights_init_naive)
        # For visualizations
        self.real = None
        self.fake = None
        self.gen_transforms = gen_transforms

        # Save arguments
        self.use_half_precision = self._configuration['use_half_precision'] and str(device) != 'cpu' \
            if 'use_half_precision' in self._configuration.keys() else False
        self.disc_iters = disc_iters if 'disc_iters' not in self._configuration else self._configuration['disc_iters']

        # Create a GradScaler once (for float16 forward/backward passes)
        if self.use_half_precision:
            self.grad_scaler = GradScaler()

    def _init_gen_disc_opt_scheduler(self, resolution: int = 4, num_iters: Optional[int] = None,
                                     batch_size: Optional[int] = None, device: Optional[str or torch.device] = None):
        """
        Initialize Generator and Discriminator networks.
        :param (int) resolution: height and width of current generation/discrimination
        :param (int) num_iters: number of iterations the new networks will run
        :param (int) batch_size: batch_size of the new networks' dataloader
        :param (str or torch.device) device: the device to move networks to
        """
        if self.gen and resolution == self.gen.resolution:
            return
        self.logger.info(f'[GROWING REQUEST] resolution={resolution}, num_iters={num_iters}, batch_size={batch_size}')
        # Find number of iters
        if num_iters is None:
            if self._configuration['grow_scheduler'] and self.evaluator and self.evaluator.dataset:
                resolution_index = int(math.log2(resolution)) - 2
                batch_size = self._configuration['grow_scheduler']['batch_sizes'][resolution_index]
                transition_epochs = [None]
                transition_epochs.extend(self._configuration['grow_scheduler']['transition_epochs'])
                transition_epoch = transition_epochs[resolution_index]
                if batch_size is None or transition_epoch is None:
                    num_iters = 2
                else:
                    num_iters = (self.dataset_len // batch_size) * transition_epoch
            else:
                batch_size = None
                num_iters = 2
        self.logger.debug(f'[_init_gen_disc_opt_scheduler] num_iters:{num_iters} | batch_size={batch_size}')
        #   - Free current networks
        if self.gen is not None:
            del self.gen
            del self.gen_opt
            del self.disc
            del self.disc_opt
            gc.collect()
            time.sleep(1.0)
        #   - Generator
        self.gen = StyleGanGenerator(c_out=self._configuration['shapes']['c_out'], resolution=resolution,
                                     num_iters=num_iters, logger=self.logger, **self._configuration['gen'])
        ################################################################################################################
        ################################################# DEV LOGGING ##################################################
        ################################################################################################################
        # self.gen.plot_alpha_curve()
        ################################################################################################################
        #   - Discriminator
        self.disc = StyleGanDiscriminator(c_in=self._configuration['shapes']['c_in'], resolution=resolution,
                                          num_iters=num_iters, logger=self.logger, **self._configuration['disc'])
        ################################################################################################################
        ################################################# DEV LOGGING ##################################################
        ################################################################################################################
        # self.disc.plot_alpha_curve()
        ################################################################################################################
        #   - Move models to GPU
        if device is not None:
            self.gen.to(device)
            self.disc.to(device)
        #   - Optimizers & LR Schedulers
        self.gen_opt, self.gen_opt_lr_scheduler = get_optimizer(self.gen, **self._configuration['gen_opt'])
        self.disc_opt, self.disc_opt_lr_scheduler = get_optimizer(self.disc, **self._configuration['disc_opt'])
        # Save dataset parameters
        self.current_batch_size = batch_size
        # Reset GDriveModel counters
        self.gforward_reset()
        self.logger.info(f'[GROWING REQUEST] new_resolution: g={self.gen.resolution} d={self.disc.resolution}')

    def load_configuration(self, configuration: dict) -> None:
        IGanGModule.load_configuration(self, configuration)

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
        if state_dict["resolution"] != self.gen.resolution:
            self.logger.warning(f'state_dict["resolution"]={state_dict["resolution"]} IS NOT self.gen.resolution=' +
                                str(self.gen.resolution) + '. Re-initializing disc/gen.')
            self._init_gen_disc_opt_scheduler(resolution=state_dict['resolution'], device=self.device)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen.alpha_curve = state_dict['gen_alpha']['curve']
        self.gen.alpha_index = state_dict['gen_alpha']['index']
        self.gen.alpha = state_dict['gen_alpha']['value']
        self.disc.load_state_dict(state_dict['disc'])
        self.disc.alpha_curve = state_dict['disc_alpha']['curve']
        self.disc.alpha_index = state_dict['disc_alpha']['index']
        self.disc.alpha = state_dict['disc_alpha']['value']
        self.gen_opt.load_state_dict(state_dict['gen_opt'])
        self.disc_opt.load_state_dict(state_dict['disc_opt'])
        self.gen_losses_permanent = state_dict['gen_losses']
        self.gen_losses_indices = state_dict['gen_losses_indices']
        self.disc_losses_permanent = state_dict['disc_losses']
        self.disc_losses_indices = state_dict['disc_losses_indices']

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
            'resolution': self.gen.resolution,
            'gen': self.gen.state_dict(),
            'gen_alpha': {
                'curve': self.gen.alpha_curve,
                'index': self.gen.alpha_index,
                'value': self.gen.alpha,
            },
            'gen_loss': mean_gen_loss,
            'gen_losses': self.gen_losses_permanent.copy(),
            'gen_losses_indices': self.gen_losses_indices.copy(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc': self.disc.state_dict(),
            'disc_alpha': {
                'curve': self.disc.alpha_curve,
                'index': self.disc.alpha_index,
                'value': self.disc.alpha,
            },
            'disc_loss': mean_disc_loss,
            'disc_losses': self.disc_losses_permanent.copy(),
            'disc_losses_indices': self.disc_losses_indices.copy(),
            'disc_opt': self.disc_opt.state_dict(),
            'nparams': self.nparams,
            'nparams_hr': self.nparams_hr,
            'config_id': self.config_id,
            'configuration': self._configuration,
        }

    def forward(self, real: torch.Tensor) -> Tuple[torch.Tensor or None, torch.Tensor or None]:
        """
        Forward pass through the entire CycleGAN model.
        :param (torch.Tensor) real: batch of real images
        :return: a tuple containing: 1) discriminator loss and 2) generator loss
        """
        # Update gdrive model state
        batch_size = real.shape[0]
        if self.is_master_device:
            self.gforward(batch_size)

        # FIX: Due to the MiniBatchStd layer, batch_size must be divided by group_size (which defaults to 4)
        batch_size_mod_4 = batch_size % 4
        if batch_size_mod_4 != 0:
            batch_size -= batch_size_mod_4
            if batch_size < 4:
                return None, None
            real = real[0:batch_size]

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        with self.gen.frozen():
            disc_losses = []
            for _ in range(self.disc_iters):
                # Update discriminators
                #   - zero-out discriminators' gradients (before backprop)
                self.disc_opt.zero_grad()
                #   - produce fake images & loss using half-precision (float16)
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    fake = self.gen(self.gen.get_noise(batch_size=batch_size, device=real.device))
                    #   - compute discriminator loss
                    disc_loss = self.disc.get_loss(real=real, fake=fake)
                #   - backprop & update weights
                if self.use_half_precision:
                    self.grad_scaler.scale(disc_loss).backward()
                    self.grad_scaler.step(self.disc_opt)
                else:
                    disc_loss.backward(retain_graph=True)
                    self.disc_opt.step()
                #   - update LR (if needed)
                if self.disc_opt_lr_scheduler:
                    if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                        self.disc_opt_lr_scheduler.step(metrics=disc_loss)
                    else:
                        self.disc_opt_lr_scheduler.step()
                disc_losses.append(disc_loss.detach())
                ########################################################################################################
                ############################################# DEV LOGGING ##############################################
                ########################################################################################################
                # disc_losses.append(torch.tensor(0.0))
                ########################################################################################################
            disc_loss = torch.mean(torch.stack(disc_losses))

        ##########################################
        ########     Update Generator     ########
        ##########################################
        with self.disc.frozen():
            #   - zero-out generators' gradients
            self.gen_opt.zero_grad()
            #   - produce fake images & generator loss using half-precision (float16)
            with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                gen_loss, _ = self.gen.get_loss(batch_size=batch_size, disc=self.disc)
            #   - backprop & update weights
            if self.use_half_precision:
                self.grad_scaler.scale(gen_loss).backward()
                self.grad_scaler.step(self.gen_opt)
            else:
                gen_loss.backward()
                self.gen_opt.step()
            #   - update LR (if needed)
            if self.gen_opt_lr_scheduler:
                if isinstance(self.gen_opt_lr_scheduler, ReduceLROnPlateau):
                    self.gen_opt_lr_scheduler.step(metrics=gen_loss)
                else:
                    self.gen_opt_lr_scheduler.step()

        # Update grad scaler
        if self.use_half_precision:
            self.grad_scaler.update()

        # Update alphas
        if self.gen.alpha_index >= len(self.gen.alpha_curve):
            self.gen.alpha = 1.0
            self.disc.alpha = 1.0
        else:
            self.gen.alpha = self.gen.alpha_curve[self.gen.alpha_index]
            self.gen.alpha_index += 1
            self.disc.alpha = self.disc.alpha_curve[self.disc.alpha_index]
            self.disc.alpha_index += 1
            ############################################################################################################
            ############################################### DEV LOGGING ################################################
            ############################################################################################################
            # self.logger.debug(f'self.disc.alpha_index={self.disc.alpha_index}')
            ############################################################################################################

        # Save for visualization
        if self.is_master_device:
            self.fake = fake[::batch_size // 2 - 1].detach().cpu()
            self.real = real[::batch_size // 2 - 1].detach().cpu()
            self.gen_losses.append(gen_loss.detach().item())
            self.gen_losses_permanent.append(gen_loss.detach().item())
            self.gen_losses_indices.append(self.epoch)
            self.disc_losses.extend([dl.item() for dl in disc_losses])
            self.disc_losses_permanent.extend([dl.item() for dl in disc_losses])
            self.disc_losses_indices.extend([self.epoch for _ in disc_losses])

        return disc_loss, gen_loss.detach()

    def growing(self) -> bool:
        """
        Function to call on every step to check if the network should grow.
        :return: True if networks grew during function call, False if no change occurred
        """
        scheduler = self._configuration['grow_scheduler']
        starting_epochs = scheduler['starting_epochs']
        batch_sizes = scheduler['batch_sizes']
        transition_epochs = [None]
        transition_epochs.extend(scheduler['transition_epochs'])

        current_epoch = self.epoch
        current_resolution = self.gen.resolution
        new_resolution = 4
        new_resolution_index = 0
        for se in reversed(range(2, int(math.log2(self._configuration['resolutions']['max'])))):
            if current_epoch >= starting_epochs[se - 2]:
                new_resolution = 2 ** (se + 1)
                new_resolution_index = se - 1
                break
        if new_resolution > current_resolution:
            batch_size = batch_sizes[new_resolution_index]
            transition_epoch = transition_epochs[new_resolution_index]
            num_iters = (len(self.evaluator.dataset) // batch_size) * transition_epoch
            self.logger.info(f'Growing network: {current_resolution} --> {new_resolution} (num_iters={num_iters})')
            self._init_gen_disc_opt_scheduler(resolution=new_resolution, num_iters=num_iters, batch_size=batch_size,
                                              device=self.device)
            self.logger.debug(f'Growing network: [DONE]')
            return True
        return False

    #
    # --------------
    # Visualizable
    # -------------
    #

    def visualize_indices(self, indices: int or Sequence) -> Image:
        # Fetch images
        assert hasattr(self, 'evaluator') and hasattr(self.evaluator, 'dataset'), 'Could not find dataset from model'
        real_images = []
        with self.gen.frozen():
            fake_images = self.gen(self.gen.get_noise(batch_size=3, device=self.device)).detach().cpu()
            for index in indices:
                real_images.append(self.evaluator.dataset[index].cpu())

        # Resize
        if self.gen.resolution != real_images[0].shape[-1]:
            fake_images = nn.Upsample(size=real_images[0].shape[-1])(fake_images)

        # Convert to grid of images
        ncols = 3
        nrows = 2
        grid = create_img_grid(images=torch.stack([
            real_images[0], real_images[1], real_images[2],
            fake_images[0], fake_images[1], fake_images[2],
        ]), ncols=ncols, gen_transforms=self.gen_transforms)

        # Plot
        return plot_grid(grid=grid, figsize=(ncols, nrows),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)} | '
                                    f'i={indices}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses), 3))}, ' +
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses), 3))}')

    def visualize(self, reproducible: bool = False) -> Image:
        if reproducible:
            return self.visualize_indices(indices=self.reproducible_indices)

        # Get first & last sample from saved images in self
        real_0 = self.real[0]
        fake_0 = self.fake[0]
        real_1 = self.real[1]
        fake_1 = self.fake[1]
        real__1 = self.real[-1]
        fake__1 = self.fake[-1]

        # Concat images to a 2x3 grid (each column is a separate generation, the rows contain real and generated images
        # respectively)
        ncols = 3
        nrows = 2
        upsample = nn.Upsample(size=self._configuration['resolutions']['max'])
        grid = create_img_grid(images=upsample(torch.stack([
            real_0, real_1, real__1,
            fake_0, fake_1, fake__1,
        ])), ncols=ncols, gen_transforms=self.gen_transforms)

        # Plot
        return plot_grid(grid=grid.numpy(), figsize=(ncols, nrows),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses), 3))}, ' +
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses), 3))}')


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
    _gen_transforms = FISBDataset.get_image_transforms(
        target_shape=StyleGan.DefaultConfiguration['resolutions']['max'],
        target_channels=StyleGan.DefaultConfiguration['shapes']['c_out']
    )

    _bs = 4
    _dl = FISBDataloader(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                         log_level=_log_level, batch_size=_bs, pin_memory=False)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_root, gen_dataset=_dl.dataset, z_dim=512,
                              n_samples=4, batch_size=2, f1_k=1, device='cpu')

    # Initialize model
    _stgan = StyleGan(model_fs_folder_or_root=_models_root, config_id='default', chkpt_step=None, chkpt_epoch=None,
                      dataset_len=len(_dl.dataset), log_level=_log_level, evaluator=_evaluator, device='cpu')
    # _stgan._init_gen_disc_opt_scheduler(resolution=128)
    # _stgan.use_half_precision = False
    # print(_stgan.nparams_hr)
    #
    # _device = _stgan.device
    # _x = next(iter(_dl))
    # _disc_loss, _gen_loss = _stgan(_x.to(_device))
    # print(_disc_loss, _gen_loss)
    #
    # _state_dict = _stgan.state_dict()
    # torch.save(_state_dict, '/home/achariso/PycharmProjects/gans-thesis/src/checkpoint.pth')

    _metrics = _evaluator.evaluate(gen=_stgan.gen, show_progress=True)
    exit(0)

    _img = _stgan.visualize()
    with open('/home/achariso/Pictures/Thesis/stgan_sample.png', 'wb') as _img_fp:
        _img.save(_img_fp)
        os.system('xdg-open /home/achariso/Pictures/Thesis/stgan_sample.png')
    plt.imshow(_img)
    plt.show()
