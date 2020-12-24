from typing import Tuple, Optional

import torch.nn as nn
from torch import Tensor, no_grad

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator
from utils.train import get_adam_optimizer, get_optimizer_lr_scheduler


class CycleGAN(nn.Module):
    """
    CycleGAN Class:
    This is the whole CycleGAN model consisting of two pix2pixHD-like Generators and two PatchGAN Discriminators each
    for its respective domain.
    """

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
                                            n_residual_blocks=n_residual_blocks_gen)
        self.gen_b_to_a = CycleGANGenerator(c_in=c_in, c_out=c_out, c_hidden=c_hidden_gen,
                                            n_residual_blocks=n_residual_blocks_gen)
        # Domain Discriminators
        self.disc_a = PatchGANDiscriminator(c_in=c_in, c_hidden=c_hidden_disc,
                                            n_contracting_blocks=n_contracting_blocks_disc,
                                            use_spectral_norm=use_spectral_norm_disc)
        self.disc_b = PatchGANDiscriminator(c_in=c_in, c_hidden=c_hidden_disc,
                                            n_contracting_blocks=n_contracting_blocks_disc,
                                            use_spectral_norm=use_spectral_norm_disc)

        # Optimizers
        self.gen_opt = get_adam_optimizer(self.gen_a_to_b, self.gen_b_to_a, lr=1e-2)
        self.disc_a_opt = get_adam_optimizer(self.disc_a, lr=1e-2)
        self.disc_b_opt = get_adam_optimizer(self.disc_b, lr=1e-2)
        # Optimizer LR Schedulers
        self.lr_scheduler_type = lr_scheduler_type
        if lr_scheduler_type is not None:
            self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(self.gen_opt, schedule_type=lr_scheduler_type)
            self.disc_a_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_a_opt, schedule_type=lr_scheduler_type)
            self.disc_b_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_b_opt, schedule_type=lr_scheduler_type)

    def opt_zero_grad(self) -> None:
        """
        Erases previous gradients for all optimizers defined in this model.
        """
        self.disc_a_opt.zero_grad()
        self.disc_b_opt.zero_grad()
        self.gen_opt.zero_grad()

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
        # Erase previous gradients
        self.opt_zero_grad()
        # Produce fake images for discriminators
        with no_grad():
            fake_a = self.gen_b_to_a(real_b)
            fake_b = self.gen_a_to_b(real_a)
        # Update discriminator A
        disc_a_loss = self.disc_a.get_loss(real=real_a, fake=fake_a, criterion=nn.MSELoss())
        disc_a_loss.backward(retain_graph=True)
        self.disc_a_opt.step()
        # Update discriminator B
        disc_b_loss = self.disc_b.get_loss(real=real_b, fake=fake_b, criterion=nn.MSELoss())
        disc_b_loss.backward(retain_graph=True)
        self.disc_b_opt.step()
        # Update generators
        gen_loss, fake_a, fake_b = self.get_gen_loss(real_a, real_b, lambda_identity=lambda_identity,
                                                     lambda_cycle=lambda_cycle)
        gen_loss.backward()
        self.gen_opt.step()
        # Update LR (if needed)
        if self.lr_scheduler_type is not None:
            self.disc_a_opt_lr_scheduler.step(metrics=disc_a_loss) if self.lr_scheduler_type == 'on_plateau' \
                else self.disc_a_opt_lr_scheduler.step()
            self.disc_b_opt_lr_scheduler.step(metrics=disc_b_loss) if self.lr_scheduler_type == 'on_plateau' \
                else self.disc_b_opt_lr_scheduler.step()
            self.gen_opt_lr_scheduler.step(metrics=gen_loss) if self.lr_scheduler_type == 'on_plateau' \
                else self.gen_opt_lr_scheduler.step()
        return disc_a_loss, disc_b_loss, gen_loss, fake_a, fake_b

    def get_gen_loss(self, real_a: Tensor, real_b: Tensor, adv_criterion: nn.modules.Module = nn.MSELoss(),
                     identity_criterion: nn.modules.Module = nn.L1Loss(),
                     cycle_criterion: nn.modules.Module = nn.L1Loss(),
                     lambda_identity: float = 0.1,
                     lambda_cycle: float = 10) -> Tuple[Tensor, Tensor, Tensor]:
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
