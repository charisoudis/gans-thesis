from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.discriminators.munit import MUNITDiscriminator
from modules.generators.munit import MUNITGenerator
from utils.train import get_optimizer, get_optimizer_lr_scheduler


class MUNIT(nn.Module):
    """
    MUNIT Class:
    This is the entire MUNIT model consisting of two AE-like Generators and two (stacks of) PatchGAN Discriminators each
    for its respective domain.
    """

    def __init__(self, gen_channels: int = 64, n_content_downsampling_blocks: int = 2,
                 n_style_downsampling_blocks: int = 4, n_residual_blocks: int = 4, s_dim: int = 8, h_dim: int = 256,
                 disc_hidden_channels: int = 64, n_disc_contracting_blocks: int = 3, n_discriminators: int = 3,
                 lr_scheduler_type: Optional[str] = None):
        """
        MUNIT class constructor.
        :param gen_channels: number of channels in first conv layers of generators encoders
        :param n_content_downsampling_blocks: number of Downsampling blocks in generators ContentEncoders
        :param n_style_downsampling_blocks: number of Downsampling blocks in generators ContentEncoders
        :param n_residual_blocks: number of residual blocks in generators encoder
        :param s_dim: length of style vectors in generators
        :param h_dim: number of hidden neurons in style transformers in generators
        :param disc_hidden_channels: number of hidden channels in PatchGAN discriminators
        :param n_disc_contracting_blocks: number of contracting blocks in PatchGAN discriminators
        :param n_discriminators: number of PatchGAN discriminators
        :param lr_scheduler_type: if specified, optimizers use LR scheduling of given type
        """
        super().__init__()
        # Domain Generators
        self.gen_a = MUNITGenerator(gen_channels, n_content_downsampling_blocks, n_style_downsampling_blocks,
                                    n_residual_blocks, s_dim, h_dim)
        self.gen_b = MUNITGenerator(gen_channels, n_content_downsampling_blocks, n_style_downsampling_blocks,
                                    n_residual_blocks, s_dim, h_dim)
        # Domain Discriminators
        self.disc_a = MUNITDiscriminator(n_discriminators, c_in=3, c_hidden=disc_hidden_channels,
                                         n_contracting_blocks=n_disc_contracting_blocks)
        self.disc_b = MUNITDiscriminator(n_discriminators, c_in=3, c_hidden=disc_hidden_channels,
                                         n_contracting_blocks=n_disc_contracting_blocks)
        self.s_dim = s_dim

        # Optimizers
        # Attention: In this version of MUNIT we jointly train both generators and jointly train both discriminators
        # noinspection PyTypeChecker
        self.gen_opt, _ = get_optimizer(self.gen_a, self.gen_b, lr=1e-2)
        # noinspection PyTypeChecker
        self.disc_opt, _ = get_optimizer(self.disc_a, self.disc_b, lr=1e-2)
        # Optimizer LR Schedulers
        self.lr_scheduler_type = lr_scheduler_type
        if lr_scheduler_type is not None:
            self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(self.gen_opt, schedule_type=lr_scheduler_type)
            self.disc_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_opt, schedule_type=lr_scheduler_type)

    def opt_zero_grad(self) -> None:
        """
        Erases previous gradients for all optimizers defined in this model.
        """
        self.disc_opt.zero_grad()
        self.gen_opt.zero_grad()

    def forward(self, x_a: Tensor, x_b: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Function for completing a forward pass of MUNIT Generator + Discriminator:
        Given an image tensor for each of the two domains A, B, it passes them through the Generators and computes
        Autoencoder losses (image & cross-domain latent code reconstruction). Also uses Discriminator to compute
        adversarial losses for Generators as well as for the Discriminators. Returns generator joint loss, discriminator
        joint loss and cross-domain generated image (x_xy = content from domain X + style from domain Y).
        :param x_a: tensor of real images from domain A
        :param x_b: tensor of real images from domain A
        :return: a tuple containing: 1) Generators (joint) loss, 2) Discriminators (joint) loss, 3) Images from domain
        A with style applied from domain B and 4) the reciprocal
        """
        # Erase previous gradients
        self.opt_zero_grad()
        # In fig. 2 of MUNIT paper, they sample style of domain A & B from normal distributions and use those to compute
        # the Autoencoder losses
        s_a = torch.randn(x_a.shape[0], self.s_dim, 1, 1, device=x_a.device).to(x_a.dtype)
        s_b = torch.randn(x_b.shape[0], self.s_dim, 1, 1, device=x_b.device).to(x_b.dtype)
        # Encode real x and compute image reconstruction loss
        x_a_loss, c_a, s_a_fake = self.gen_a.ae_image_recon_loss(x_a)
        x_b_loss, c_b, s_b_fake = self.gen_b.ae_image_recon_loss(x_b)
        # Decode real (c, s) and compute latent reconstruction loss
        c_b_loss, s_a_loss, x_ba = self.gen_a.ae_latent_recon_loss(c_b, s_a)
        c_a_loss, s_b_loss, x_ab = self.gen_b.ae_latent_recon_loss(c_a, s_b)
        # Compute adversarial losses
        with torch.no_grad():
            gen_a_adv_loss = self.disc_a.get_loss(x_ba, is_real=True)
            gen_b_adv_loss = self.disc_b.get_loss(x_ab, is_real=True)
        # Update discriminators
        disc_loss = (
            # Discriminator for domain A
            self.disc_a.get_loss(x_ba.detach(), is_real=False) +
            self.disc_a.get_loss(x_a.detach(), is_real=True) +
            # Discriminator for domain B
            self.disc_b.get_loss(x_ab.detach(), is_real=False) +
            self.disc_b.get_loss(x_b.detach(), is_real=True)
        )
        disc_loss.backward(retain_graph=True)
        self.disc_opt.step()
        # Update generators
        gen_loss = (
                10 * x_a_loss + c_a_loss + s_a_loss + gen_a_adv_loss +
                10 * x_b_loss + c_b_loss + s_b_loss + gen_b_adv_loss
        )
        gen_loss.backward()
        self.gen_opt.step()
        # Update LR (if needed)
        if self.lr_scheduler_type is not None:
            self.disc_opt_lr_scheduler.step(metrics=disc_loss) if self.lr_scheduler_type == 'on_plateau' \
                else self.disc_opt_lr_scheduler.step()
            self.gen_opt_lr_scheduler.step(metrics=gen_loss) if self.lr_scheduler_type == 'on_plateau' \
                else self.gen_opt_lr_scheduler.step()

        assert isinstance(gen_loss, Tensor)
        assert isinstance(disc_loss, Tensor)
        return gen_loss, disc_loss, x_ab, x_ba
