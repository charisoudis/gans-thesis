from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator
from modules.ifaces import IGanGModule
from utils.train import get_adam_optimizer


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
            },
            'gen_b_to_a': {
                'c_hidden': 64,
                'n_contracting_blocks': 2,
                'n_residual_blocks': 9,
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
            'disc_a': {
                'c_hidden': 8,
                'n_contracting_blocks': 4,
                'n_residual_blocks': 9,
            },
            'disc_b': {
                'c_hidden': 8,
                'n_contracting_blocks': 4,
                'n_residual_blocks': 9,
            },
            'use_spectral_norm': True,
            'recon_criterion': 'L1',
            'adv_criterion': 'MSE',
        },
        'disc_opt': {
            'lr': 1e-4,
            'opt': 'adam',
            'scheduler': None
        }
    }

    def __init__(self, c_in: int = 3, c_out: int = 3, c_hidden_gen: int = 64, n_residual_blocks_gen: int = 9,
                 c_hidden_disc: int = 8, n_contracting_blocks_disc: int = 4, use_spectral_norm_disc: bool = False,
                 lr_scheduler_type: Optional[str] = None):
        """
        CycleGAN class constructor.
        :param c_in: the number of channels to expect from a given input
        :param c_out: the number of channels to expect for a given output
        :param c_hidden_gen: the number of hidden channels in generators (e.g. in Residual blocks)
        :param n_residual_blocks_gen: the number of Residual blocks used in each generator
        :param c_hidden_disc: the number of hidden channels in discriminators
        :param n_contracting_blocks_disc: the number of contracting blocks in discriminators
        :param use_spectral_norm_disc: if use spectral_norm (to penalize weight gradients) in discriminators
        :param lr_scheduler_type: if specified, optimizers use LR scheduling of given type (supported: 'on_plateau',
                                  'cyclic', )
        """
        super(CycleGAN, self).__init__()
        # Domain Generators
        self.gen_a_to_b = CycleGANGenerator(c_in=c_in, c_out=c_out, c_hidden=c_hidden_gen,
                                            n_residual_blocks=n_residual_blocks_gen, n_contracting_blocks=2)
        self.gen_b_to_a = CycleGANGenerator(c_in=c_in, c_out=c_out, c_hidden=c_hidden_gen,
                                            n_residual_blocks=n_residual_blocks_gen)
        # Domain Discriminators
        self.disc_a = PatchGANDiscriminator(c_in=c_in, c_hidden=c_hidden_disc,
                                            n_contracting_blocks=n_contracting_blocks_disc,
                                            use_spectral_norm=use_spectral_norm_disc)
        self.disc_b = PatchGANDiscriminator(c_in=c_in, c_hidden=c_hidden_disc,
                                            n_contracting_blocks=n_contracting_blocks_disc,
                                            use_spectral_norm=use_spectral_norm_disc)

        # Optimizers & LR Schedulers
        self.gen_opt, self.gen_opt_lr_scheduler = get_adam_optimizer(self.gen_a_to_b, self.gen_b_to_a, lr=1e-4,
                                                                     weight_decay=1e-4,
                                                                     lr_scheduler_type=lr_scheduler_type)
        self.disc_opt, self.disc_opt_lr_scheduler = get_adam_optimizer(self.disc_a, self.disc_b, lr=1e-4,
                                                                       weight_decay=1e-4,
                                                                       lr_scheduler_type=lr_scheduler_type)
        self.lr_scheduler_type = lr_scheduler_type

        # For visualizations
        self.fake_a = None
        self.fake_b = None
        self.real_a = None
        self.real_b = None

        # Creates a GradScaler (for float16 forward/backward passes) once
        self.grad_scaler = GradScaler()

    def forward(self, real_a: Tensor, real_b: Tensor, lambda_identity: float = 0.1, lambda_cycle: float = 10) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the whole CycleGAN model.
        :param real_a: batch of images from real dataset of domain A
        :param real_b: batch of images from real dataset of domain B
        :param lambda_identity: weight for identity loss in total generator loss
        :param lambda_cycle: weight for cycle-consistency loss in total generator loss
        :return: a tuple containing: 1) discriminator of domain A loss, 2) discriminator of domain B loss, 3) Joint
                 generators loss, 4) fake images from generator B --> A and 5) fake images from generator A --> B
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
                self.disc_opt.zero_grad()
                #   - produce fake images & loss using half-precision (float16)
                with torch.cuda.amp.autocast():
                    fake_b = self.gen_a_to_b(real_a)
                    fake_a = self.gen_b_to_a(real_b)
                    #   - compute joint discriminator loss
                    disc_a_loss = self.disc_a.get_loss(real=real_a, fake=fake_a, criterion=nn.MSELoss())
                    disc_b_loss = self.disc_b.get_loss(real=real_b, fake=fake_b, criterion=nn.MSELoss())
                    disc_loss = 0.5 * (disc_a_loss + disc_b_loss)
                #   - backprop & update weights
                # disc_loss.backward(retain_graph=True)
                # self.disc_opt.step()
                self.grad_scaler.scale(disc_loss).backward()
                self.grad_scaler.step(self.disc_opt)
                #   - update LR (if needed)
                if self.disc_opt_lr_scheduler:
                    if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                        self.disc_opt_lr_scheduler.step(metrics=disc_loss)
                    else:
                        self.disc_opt_lr_scheduler.step()

        #############################################
        ########     Update Generator(s)     ########
        #############################################
        with self.disc_a.frozen():
            with self.disc_b.frozen():
                #   - zero-out generators' gradients
                self.gen_opt.zero_grad()
                #   - produce fake images & generator loss using half-precision (float16)
                with torch.cuda.amp.autocast():
                    gen_loss, fake_a, fake_b = self.get_gen_loss(real_a=real_a, real_b=real_b,
                                                                 adv_criterion=nn.MSELoss(),
                                                                 lambda_identity=lambda_identity,
                                                                 lambda_cycle=lambda_cycle)
                #   - backprop & update weights
                # gen_loss.backward()
                # self.gen_opt.step()
                self.grad_scaler.scale(gen_loss).backward()
                self.grad_scaler.step(self.gen_opt)
                #   - update LR (if needed)
                if self.gen_opt_lr_scheduler:
                    if isinstance(self.gen_opt_lr_scheduler, ReduceLROnPlateau):
                        self.gen_opt_lr_scheduler.step(metrics=gen_loss)
                    else:
                        self.gen_opt_lr_scheduler.step()

        # Update grad scaler
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
