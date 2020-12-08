from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.discriminators.munit import MUNITDiscriminator
from modules.generators.munit import MUNITGenerator


class MUNIT(nn.Module):
    """
    MUNIT Class:
    This is the whole MUNIT model consisting of two AE-like Generators and two (stacks of) PatchGAN Discriminators each
    for its respective domain.
    """

    def __init__(self, gen_channels: int = 64, n_content_downsampling_blocks: int = 2,
                 n_style_downsampling_blocks: int = 4, n_residual_blocks: int = 4, s_dim: int = 8, h_dim: int = 256,
                 disc_hidden_channels: int = 64, n_disc_contracting_blocks: int = 3, n_discriminators: int = 3):
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
        # In fig. 2 of MUNIT paper the sample style of domain A & B from normal distributions and use those to compute
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
        gen_a_adv_loss = self.disc_a.get_loss(x_ba, is_real=True)
        gen_b_adv_loss = self.disc_b.get_loss(x_ab, is_real=True)
        # Sum up losses for generator
        gen_loss = (
                10 * x_a_loss + c_a_loss + s_a_loss + gen_a_adv_loss +
                10 * x_b_loss + c_b_loss + s_b_loss + gen_b_adv_loss
        )
        # Sum up losses for discriminator
        disc_loss = (
            # Discriminator for domain A
            self.disc_a.get_loss(x_ba.detach(), is_real=False) +
            self.disc_a.get_loss(x_a.detach(), is_real=True) +
            # Discriminator for domain B
            self.disc_b.get_loss(x_ab.detach(), is_real=False) +
            self.disc_b.get_loss(x_b.detach(), is_real=True)
        )
        return gen_loss, disc_loss, x_ab, x_ba
