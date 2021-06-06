from typing import Tuple, Optional, Sequence

import click
import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from datasets.bags2shoes import Bags2ShoesDataset, Bags2ShoesDataloader
from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator
from modules.ifaces import IGanGModule
from utils.filesystems.local import LocalCapsule, LocalFolder
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.plot import plot_grid, create_img_grid
from utils.train import get_optimizer, weights_init_naive


class CycleGAN(nn.Module, IGanGModule):
    """
    CycleGAN Class:
    This is the whole CycleGAN model consisting of two pix2pixHD-like Generators and two PatchGAN Discriminators each
    for its respective domain.
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'shapes': {
            'c_in': 3,
            'c_out': 3,
            'w_in': 64,
            'h_in': 64,
        },
        'gen': {
            'gen_a_to_b': {
                'c_hidden': 64,
                'n_contracting_blocks': 2,
                'n_residual_blocks': 9,
                'output_padding': 1,
                'use_skip_connections': False
            },
            'gen_b_to_a': {
                'c_hidden': 64,
                'n_contracting_blocks': 2,
                'n_residual_blocks': 9,
                'output_padding': 1,
                'use_skip_connections': False
            },
            'adv_criterion': 'MSE',
            'cycle_criterion': 'L1',
            'identity_criterion': 'L1',
        },
        'gen_opt': {
            'optim_type': 'Adam',
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'scheduler_type': None,
            'betas': (0.5, 0.999)
        },
        'disc': {
            'disc_a': {
                'c_hidden': 64,
                'n_contracting_blocks': 3,
                'use_spectral_norm': True,
                'adv_criterion': 'MSE',
            },
            'disc_b': {
                'c_hidden': 64,
                'n_contracting_blocks': 3,
                'use_spectral_norm': True,
                'adv_criterion': 'MSE',
            },
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
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id=config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, reproducible_indices=reproducible_indices,
                             evaluator=evaluator, **evaluator_kwargs)

        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define CycleGAN model
        shapes_conf = self._configuration['shapes']
        criteria_conf = {
            'adv': getattr(nn, f'{self._configuration["gen"]["adv_criterion"]}Loss')(),
            'cycle': getattr(nn, f'{self._configuration["gen"]["cycle_criterion"]}Loss')(),
            'identity': getattr(nn, f'{self._configuration["gen"]["identity_criterion"]}Loss')(),
        }
        #   - Domain Generators
        self.gen_a_to_b = CycleGANGenerator(c_in=shapes_conf['c_in'], c_out=shapes_conf['c_out'],
                                            logger=self.logger, criteria_conf=criteria_conf,
                                            **self._configuration['gen']['gen_a_to_b'])
        self.gen_b_to_a = CycleGANGenerator(c_in=shapes_conf['c_in'], c_out=shapes_conf['c_out'],
                                            logger=self.logger, criteria_conf=criteria_conf,
                                            **self._configuration['gen']['gen_b_to_a'])
        #   - Domain Discriminators
        self.disc_a = PatchGANDiscriminator(c_in=shapes_conf['c_in'], logger=self.logger,
                                            **self._configuration['disc']['disc_a'])
        self.disc_b = PatchGANDiscriminator(c_in=shapes_conf['c_in'], logger=self.logger,
                                            **self._configuration['disc']['disc_b'])
        # Move models to {C,G,T}PU
        self.gen_a_to_b.to(device)
        self.gen_b_to_a.to(device)
        self.disc_a.to(device)
        self.disc_b.to(device)
        self.device = device
        self.is_master_device = (isinstance(device, torch.device) and device.type == 'cuda' and device.index == 0) \
                                or (isinstance(device, torch.device) and device.type == 'cpu') \
                                or (isinstance(device, str) and device == 'cpu')
        #   - Optimizers & LR Schedulers
        self.gen_opt, self.gen_opt_lr_scheduler = get_optimizer(self.gen_a_to_b, self.gen_b_to_a,
                                                                **self._configuration['gen_opt'])
        disc_opt_conf = self._configuration['disc_opt']
        self.disc_joint_opt = 'joint_opt' not in disc_opt_conf.keys() or disc_opt_conf['joint_opt'] is True
        del disc_opt_conf['joint_opt']
        if self.disc_joint_opt:
            self.disc_opt, self.disc_opt_lr_scheduler = get_optimizer(self.disc_a, self.disc_b, **disc_opt_conf)
        else:
            self.disc_a_opt, self.disc_a_opt_lr_scheduler = get_optimizer(self.disc_a, **disc_opt_conf)
            self.disc_b_opt, self.disc_b_opt_lr_scheduler = get_optimizer(self.disc_b, **disc_opt_conf)

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

        # Create a GradScaler (for float16 forward/backward passes) once
        self.grad_scaler = GradScaler()

        # Save arguments
        self.device = device
        self.use_half_precision = self._configuration['use_half_precision'] \
            if 'use_half_precision' in self._configuration.keys() else False

    def load_configuration(self, configuration: dict) -> None:
        IGanGModule.load_configuration(self, configuration)

    @property
    def gen(self) -> nn.Module:
        return self.gen_a_to_b

    @gen.setter
    def gen(self, gen: Optional[nn.Module]) -> None:
        pass

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
                                 f'"{state_dict["config_id"]}"). SHOULD NOT apply checkpoint...')
            if not click.confirm('Override config_id in checkpoint and attempt to load it?', default=False):
                return
        # Load model checkpoints
        # FIX: Keys
        if not self.gen_a_to_b.use_skip_connections:
            if 'expand0.expanding_block.0.weight' in state_dict['gen_a_to_b'].keys():
                del state_dict['gen_a_to_b']['expand0.expanding_block.0.weight']
            if 'expand0.expanding_block.0.bias' in state_dict['gen_a_to_b'].keys():
                del state_dict['gen_a_to_b']['expand0.expanding_block.0.bias']
            if 'expand1.expanding_block.0.weight' in state_dict['gen_a_to_b'].keys():
                del state_dict['gen_a_to_b']['expand1.expanding_block.0.weight']
            if 'expand1.expanding_block.0.bias' in state_dict['gen_a_to_b'].keys():
                del state_dict['gen_a_to_b']['expand1.expanding_block.0.bias']
        if not self.gen_b_to_a.use_skip_connections:
            if 'expand0.expanding_block.0.weight' in state_dict['gen_b_to_a'].keys():
                del state_dict['gen_b_to_a']['expand0.expanding_block.0.weight']
            if 'expand0.expanding_block.0.bias' in state_dict['gen_b_to_a'].keys():
                del state_dict['gen_b_to_a']['expand0.expanding_block.0.bias']
            if 'expand1.expanding_block.0.weight' in state_dict['gen_b_to_a'].keys():
                del state_dict['gen_b_to_a']['expand1.expanding_block.0.weight']
            if 'expand1.expanding_block.0.bias' in state_dict['gen_b_to_a'].keys():
                del state_dict['gen_b_to_a']['expand1.expanding_block.0.bias']
        self.gen_a_to_b.load_state_dict(state_dict['gen_a_to_b'])
        self.gen_b_to_a.load_state_dict(state_dict['gen_b_to_a'])
        try:
            self.gen_opt.load_state_dict(state_dict['gen_opt'])
        except ValueError:
            pass
        self.disc_a.load_state_dict(state_dict['disc_a'])
        self.disc_b.load_state_dict(state_dict['disc_b'])
        if self.disc_joint_opt:
            self.disc_opt.load_state_dict(state_dict['disc_opt'])
        elif 'disc_opt' in state_dict.keys():
            disc_a_opt_state_dict = state_dict['disc_opt']
            __l = len(disc_a_opt_state_dict['param_groups'][0]['params'])
            __p = state_dict['disc_opt']['param_groups'][0]['params'].copy()
            disc_a_opt_state_dict['param_groups'][0]['params'] = __p[0:__l // 2]
            self.disc_a_opt.load_state_dict(disc_a_opt_state_dict)
            disc_b_opt_state_dict = state_dict['disc_opt']
            disc_b_opt_state_dict['param_groups'][0]['params'] = __p[__l // 2:__l]
            self.disc_b_opt.load_state_dict(disc_b_opt_state_dict)
            state_dict['nparams'] = self.nparams
        else:
            self.disc_a_opt.load_state_dict(state_dict['disc_a_opt'])
            self.disc_b_opt.load_state_dict(state_dict['disc_b_opt'])
        self._nparams = state_dict['nparams']
        # Update latest metrics with checkpoint's metrics
        if 'metrics' in state_dict.keys():
            self.latest_metrics = state_dict['metrics']
        self.logger.debug(f'State dict loaded. Keys: {tuple(state_dict.keys())}')
        for _k in [_k for _k in state_dict.keys() if _k not in ('gen_a_to_b', 'gen_b_to_a', 'gen_opt', 'disc_a',
                                                                'disc_b', 'disc_opt', 'disc_a_opt', 'disc_b_opt',
                                                                'configuration')]:
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
        disc_opt_dict = {
            'disc_opt': self.disc_opt.state_dict(),
        } if self.disc_joint_opt else {
            'disc_a_opt': self.disc_a_opt.state_dict(),
            'disc_b_opt': self.disc_b_opt.state_dict(),
        }
        return {
            **{
                'gen_a_to_b': self.gen_a_to_b.state_dict(),
                'gen_b_to_a': self.gen_b_to_a.state_dict(),
                'gen_loss': mean_gen_loss,
                'gen_opt': self.gen_opt.state_dict(),
                'disc_a': self.disc_a.state_dict(),
                'disc_b': self.disc_b.state_dict(),
                'disc_loss': mean_disc_loss,
                'nparams': self.nparams,
                'nparams_hr': self.nparams_hr,
                'config_id': self.config_id,
                'configuration': self._configuration,
            },
            **disc_opt_dict
        }

    def forward(self, real_a: Tensor, real_b: Tensor, lambda_identity: float = 0.1, lambda_cycle: float = 10) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the whole CycleGAN model.
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
        with self.gen_a_to_b.frozen():
            with self.gen_b_to_a.frozen():
                # Update discriminators
                #   - zero-out discriminators' gradients (before backprop)
                if self.disc_joint_opt:
                    self.disc_opt.zero_grad()
                else:
                    self.disc_a_opt.zero_grad()
                    self.disc_b_opt.zero_grad()
                #   - produce fake images & loss using half-precision (float16)
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    fake_b = self.gen_a_to_b(real_a)
                    fake_a = self.gen_b_to_a(real_b)
                    #   - compute joint discriminator loss
                    disc_a_loss = self.disc_a.get_loss(real=real_a, fake=fake_a, criterion=nn.MSELoss())
                    disc_b_loss = self.disc_b.get_loss(real=real_b, fake=fake_b, criterion=nn.MSELoss())
                    disc_loss = 0.5 * (disc_a_loss + disc_b_loss)
                #   - backprop & update weights
                if self.device == 'cuda':
                    if self.disc_joint_opt:
                        self.grad_scaler.scale(disc_loss).backward()
                        self.grad_scaler.step(self.disc_opt)
                    else:
                        self.grad_scaler.scale(disc_a_loss).backward()
                        self.grad_scaler.step(self.disc_a_opt)
                        self.grad_scaler.scale(disc_b_loss).backward()
                        self.grad_scaler.step(self.disc_b_opt)
                else:
                    if self.disc_joint_opt:
                        disc_loss.backward(retain_graph=True)
                        self.disc_opt.step()
                    else:
                        disc_a_loss.backward(retain_graph=True)
                        self.disc_a_opt.step()
                        disc_b_loss.backward(retain_graph=True)
                        self.disc_b_opt.step()
                #   - update LR (if needed)
                if self.disc_joint_opt and self.disc_opt_lr_scheduler:
                    if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                        self.disc_opt_lr_scheduler.step(metrics=disc_loss)
                    else:
                        self.disc_opt_lr_scheduler.step()
                elif not self.disc_joint_opt:
                    if self.disc_a_opt_lr_scheduler:
                        if isinstance(self.disc_a_opt_lr_scheduler, ReduceLROnPlateau):
                            self.disc_a_opt_lr_scheduler.step(metrics=disc_loss)
                        else:
                            self.disc_a_opt_lr_scheduler.step()
                    if self.disc_b_opt_lr_scheduler:
                        if isinstance(self.disc_b_opt_lr_scheduler, ReduceLROnPlateau):
                            self.disc_b_opt_lr_scheduler.step(metrics=disc_loss)
                        else:
                            self.disc_b_opt_lr_scheduler.step()

        #############################################
        ########     Update Generator(s)     ########
        #############################################
        with self.disc_a.frozen():
            with self.disc_b.frozen():
                #   - zero-out generators' gradients
                self.gen_opt.zero_grad()
                #   - produce fake images & generator loss using half-precision (float16)
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    gen_loss, fake_a, fake_b = self.get_gen_loss(real_a=real_a, real_b=real_b,
                                                                 adv_criterion=nn.MSELoss(),
                                                                 lambda_identity=lambda_identity,
                                                                 lambda_cycle=lambda_cycle)
                #   - backprop & update weights
                if self.device == 'cuda':
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
        if self.device == 'cuda':
            self.grad_scaler.update()

        # Save for visualization
        if self.is_master_device:
            self.fake_a = fake_a[::len(fake_a) - 1].detach().cpu()
            self.fake_b = fake_b[::len(fake_b) - 1].detach().cpu()
            self.real_a = real_a[::len(real_a) - 1].detach().cpu()
            self.real_b = real_b[::len(real_b) - 1].detach().cpu()
            self.gen_losses.append(gen_loss.item())
            self.disc_losses.append(disc_loss.item())

        return disc_loss, gen_loss

        # # Erase previous gradients
        # self.gen_opt.zero_grad()
        # self.disc_opt.zero_grad()
        # # Produce fake images for discriminators
        # with no_grad():
        #     fake_a = self.gen_b_to_a(real_b)
        #     fake_b = self.gen_a_to_b(real_a)
        # # Update discriminator A
        # disc_a_loss = self.disc_a.get_loss(real=real_a, fake=fake_a, criterion=nn.MSELoss())
        # disc_a_loss.backward(retain_graph=True)
        # self.disc_a_opt.step()
        # # Update discriminator B
        # disc_b_loss = self.disc_b.get_loss(real=real_b, fake=fake_b, criterion=nn.MSELoss())
        # disc_b_loss.backward(retain_graph=True)
        # self.disc_b_opt.step()
        # # Update generators
        # gen_loss, fake_a, fake_b = self.get_gen_loss(real_a, real_b, lambda_identity=lambda_identity,
        #                                              lambda_cycle=lambda_cycle)
        # gen_loss.backward()
        # self.gen_opt.step()
        # # Update LR (if needed)
        # if self.lr_scheduler_type is not None:
        #     self.disc_a_opt_lr_scheduler.step(metrics=disc_a_loss) if self.lr_scheduler_type == 'on_plateau' \
        #         else self.disc_a_opt_lr_scheduler.step()
        #     self.disc_b_opt_lr_scheduler.step(metrics=disc_b_loss) if self.lr_scheduler_type == 'on_plateau' \
        #         else self.disc_b_opt_lr_scheduler.step()
        #     self.gen_opt_lr_scheduler.step(metrics=gen_loss) if self.lr_scheduler_type == 'on_plateau' \
        #         else self.gen_opt_lr_scheduler.step()
        # return disc_a_loss, disc_b_loss, gen_loss, fake_a, fake_b

    def get_gen_loss(self, real_a: Tensor, real_b: Tensor, adv_criterion: nn.modules.Module = nn.MSELoss(),
                     identity_criterion: nn.modules.Module = nn.L1Loss(),
                     cycle_criterion: nn.modules.Module = nn.L1Loss(),
                     lambda_identity: float = 0.1, lambda_cycle: float = 10) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get the loss of the generator given inputs.
        :param real_a: the real images from pile A
        :param real_b: the real images from pile B
        :param adv_criterion: the adversarial loss function; takes the discriminator predictions and the true labels
               and returns a adversarial loss (which we aim to minimize)
        :param identity_criterion: the reconstruction loss function used for identity loss and cycle consistency loss;
               takes two sets of images and returns their pixel differences (which we aim to minimize)
        :param cycle_criterion: the cycle consistency loss function; takes the real images from X and those images put
               through a X->Y generator and then Y->X generator and returns the cycle consistency loss (which you aim to
               minimize).
        :param lambda_identity: the weight of the identity loss
        :param lambda_cycle: the weight of the cycle-consistency loss
        :return: a tuple containing the loss (a scalar), and the generators' outputs
        """
        # Adversarial Loss
        adversarial_loss_ab, _ = self.gen_a_to_b.get_adv_loss(real_a, disc_y=self.disc_b, adv_criterion=adv_criterion)
        adversarial_loss_ba, _ = self.gen_b_to_a.get_adv_loss(real_b, disc_y=self.disc_a, adv_criterion=adv_criterion)
        # Identity Loss
        identity_loss_ba, _ = self.gen_b_to_a.get_identity_loss(real_a, identity_criterion=identity_criterion)
        identity_loss_ab, _ = self.gen_a_to_b.get_identity_loss(real_b, identity_criterion=identity_criterion)
        # Cycle-consistency Loss
        fake_b = self.gen_a_to_b(real_a)
        fake_a = self.gen_b_to_a(real_b)
        cycle_consistency_loss_aba, _ = self.gen_b_to_a.get_cycle_consistency_loss(real_a, fake_b,
                                                                                   cycle_criterion=cycle_criterion)
        cycle_consistency_loss_bab, _ = self.gen_a_to_b.get_cycle_consistency_loss(real_b, fake_a,
                                                                                   cycle_criterion=cycle_criterion)
        # Total loss
        gen_loss = adversarial_loss_ab + adversarial_loss_ba \
                   + lambda_identity * (identity_loss_ab + identity_loss_ba) \
                   + lambda_cycle * (cycle_consistency_loss_aba + cycle_consistency_loss_bab)
        return gen_loss, fake_a, fake_b

    #
    # --------------
    # Visualizable
    # -------------
    #

    def visualize_indices(self, indices: int or Sequence) -> Image:
        # Fetch images
        assert hasattr(self, 'evaluator') and hasattr(self.evaluator, 'dataset'), 'Could find dataset from model'
        real_a_images = []
        real_b_images = []
        fake_a_images = []
        fake_b_images = []
        with self.gen_a_to_b.frozen():
            with self.gen_b_to_a.frozen():
                for index in indices:
                    _real_a, _real_b = self.evaluator.dataset[index]
                    _fake_b = self.gen_a_to_b(_real_a.unsqueeze(0).to(self.device)).squeeze(0)
                    _fake_a = self.gen_b_to_a(_real_b.unsqueeze(0).to(self.device)).squeeze(0)
                    real_a_images.append(_real_a.cpu())
                    fake_a_images.append(_fake_b.cpu())
                    real_b_images.append(_real_b.cpu())
                    fake_b_images.append(_fake_a.cpu())

        real_images = real_a_images + real_b_images
        fake_images = fake_a_images + fake_b_images

        if len(real_images) <= 4:
            images = real_images + fake_images
        else:
            images = []
            for i in range(0, len(real_images) - 4):
                images.append(real_images[i])
                images.append(real_images[i + 1])
                images.append(real_images[i + 2])
                images.append(real_images[i + 3])
                images.append(fake_images[i])
                images.append(fake_images[i + 1])
                images.append(fake_images[i + 2])
                images.append(fake_images[i + 3])

        # Convert to grid of images
        ncols = 4
        nrows = int(len(images) / ncols)
        grid = create_img_grid(images=torch.stack(images), nrows=nrows, ncols=ncols,
                               gen_transforms=self.gen_transforms)

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
        real_a_0 = self.real_a[0]
        fake_b_0 = self.fake_b[0]
        real_b_0 = self.real_b[0]
        fake_a_0 = self.fake_a[0]
        real_a__1 = self.real_a[-1]
        fake_b__1 = self.fake_b[-1]
        real_b__1 = self.real_b[-1]
        fake_a__1 = self.fake_a[-1]

        # Concat images to a 4x2 grid (each row is a separate generation, the columns contain real and generated images
        # side-by-side)
        ncols = 4
        nrows = 2
        grid = create_img_grid(images=torch.stack([
            real_a_0, real_a__1, real_b_0, real_b__1,
            fake_b_0, fake_b__1, fake_a_0, fake_a__1,
        ]), ncols=ncols, gen_transforms=self.gen_transforms)

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
    _gen_transforms = Bags2ShoesDataset.get_image_transforms(
        target_shape=CycleGAN.DefaultConfiguration['shapes']['w_in'],
        target_channels=CycleGAN.DefaultConfiguration['shapes']['c_in']
    )

    _bs = 2
    _dl = Bags2ShoesDataloader(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                               log_level=_log_level, batch_size=_bs, pin_memory=False)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_root, gen_dataset=_dl.dataset, target_index=1,
                              condition_indices=(0,), n_samples=2, batch_size=1, f1_k=1, device='cpu')

    # Initialize model
    _ccgan = CycleGAN(model_fs_folder_or_root=_models_root, config_id='64_MSE_L1_L1_2_9_2_9_64_4_1e4_false_false_false',
                      dataset_len=len(_dl.dataset), log_level=_log_level, evaluator=_evaluator, device='cpu')
    print(_ccgan.nparams_hr)

    _device = _ccgan.device
    _x, _y = next(iter(_dl))
    _disc_loss, _gen_loss = _ccgan(_x.to(_device), _y.to(_device))
    print(_disc_loss, _gen_loss)
    # print('Number of parameters: gen_a_to_b=' + _ccgan.gen_a_to_b.nparams_hr)
    # print('Number of parameters: gen_b_to_a=' + _ccgan.gen_b_to_a.nparams_hr)
    # print('Number of parameters: disc_a=' + _ccgan.disc_a.nparams_hr)
    # print('Number of parameters: disc_b=' + _ccgan.disc_b.nparams_hr)
    # print('Number of parameters: TOTAL=' + _ccgan.nparams_hr)
    #
    # _img = _ccgan.visualize(reproducible=(300, 1001))
    # plt.imshow(_img)
    # plt.show()
    # # _img.show()
    # # _img.show()
    # # with open('sample.png', 'wb') as fp:
    # #     _img.save(fp)
    # #     _pxldt.logger.debug('Image saved.')
    # exit(0)
