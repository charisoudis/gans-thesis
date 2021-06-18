from typing import Tuple

import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import ChannelsProjectLayer
from modules.partial.normalization import LayerNorm2d
from modules.partial.residual import ResidualBlock


class MUNITContentEncoder(nn.Module):
    """
    MUNITContentEncoder Class:
    This is the first half of the Encoder part of MUNIT autoencoder-type generator.
    """

    def __init__(self, c_base=64, n_downsampling_blocks=2, n_residual_blocks=4):
        """
        MUNITContentEncoder class constructor.
        :param c_base: number of channels in first convolutional layer
        :param n_downsampling_blocks: number of downsampling blocks
        :param n_residual_blocks: number of residual blocks
        """
        super(MUNITContentEncoder, self).__init__()
        self.out_channels = c_base * 2 ** n_downsampling_blocks

        self.munit_content_encoder = nn.Sequential(
            # Input convolutional layer
            self.get_downsampling_block(3, c_base, kernel_size=7, stride=1, padding=3),
            # Downsampling layers
            *[self.get_downsampling_block(c_base * 2 ** i, c_base * 2 ** (i + 1)) \
              for i in range(n_downsampling_blocks)],
            # Residual blocks
            *[ResidualBlock(self.out_channels, norm_type='IN') for _ in range(n_residual_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of MUNITContentEncoder:
        Given an image tensor, completes a content encoding returns the transformed tensor (aka "content code"). This
        is a downsampled and more coarse than input image.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, self.out_channels, H_out, W_out)
        """
        return self.munit_content_encoder(x)

    @property
    def channels(self):
        return self.out_channels

    @staticmethod
    def get_downsampling_block(c_in: int, c_out: int, kernel_size: int = 4, stride: int = 2,
                               padding: int = 1, use_instance_norm: bool = True) -> nn.Sequential:
        """
        Get a block of the Downsampling part of the ContentEncoder.
        :param c_in: number of channels of input tensor to block
        :param c_out: number of channels of output tensor from block
        :param kernel_size: kernel_size parameter of Conv2d layer
        :param stride: stride parameter of Conv2d layer
        :param padding: padding parameter of Conv2d layer
        :param use_instance_norm: flag on to use/skip InstanceNorm2d before activation
        :return: torch.nn.Sequential
        """
        return nn.Sequential(
            nn.ReflectionPad2d(padding=padding),
            nn.utils.spectral_norm(
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride)
            ),
            nn.InstanceNorm2d(c_out) if use_instance_norm else nn.Identity(),
            nn.ReLU()
        )


class MUNITStyleEncoder(nn.Module):
    """
    MUNITStyleEncoder Class:
    This is the second half of the Encoder part of MUNIT autoencoder-type generator.
    """

    def __init__(self, c_base=64, n_downsampling_blocks=4, n_deepen_blocks: int = 2, s_dim=4):
        """
        MUNITContentEncoder class constructor.
        :param c_base: number of channels in first convolutional layer
        :param n_downsampling_blocks: number of downsampling blocks (total = deepen + same channels)
        :param n_deepen_blocks: number of downsampling blocks that double channels
        :param s_dim: length of output style vector
        """
        super(MUNITStyleEncoder, self).__init__()
        self.c_deepen = c_base * 2 ** n_deepen_blocks

        self.munit_style_encoder = nn.Sequential(
            # Input convolutional layer
            MUNITContentEncoder.get_downsampling_block(3, c_base, kernel_size=7, stride=1, padding=3,
                                                       use_instance_norm=False),
            # Downsampling layers
            *[MUNITContentEncoder.get_downsampling_block(c_base * 2 ** i, c_base * 2 ** (i + 1),
                                                         use_instance_norm=False) for i in range(n_deepen_blocks)],
            # Downsampling layers without channels change
            *[MUNITContentEncoder.get_downsampling_block(self.c_deepen, self.c_deepen, use_instance_norm=False) \
              for _ in range(n_downsampling_blocks - n_deepen_blocks)],
            # Output layers
            nn.AdaptiveAvgPool2d(1),  # [N, out_channels, H_n, W_n] --> [N, out_channels, 1, 1]
            ChannelsProjectLayer(self.c_deepen, s_dim)  # [N, out_channels, 1, 1] --> [N, s_dim, 1, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of MUNITStyleEncoder:
        Given an image tensor, completes a style encoding and returns a style tensor (aka "style code").
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, s_dim, 1, 1)
        """
        return self.munit_style_encoder(x)


class MUNITDecoder(nn.Module):
    """
    MUNITDecoder Class:
    This is the Decoder part of MUNIT autoencoder-type generator.
    """

    def __init__(self, c_in: int, n_upsampling_blocks: int = 2, n_residual_blocks: int = 4, s_dim: int = 8,
                 h_dim: int = 256):
        """
        MUNITDecoder class constructor.
        :param c_in: number of channels from encoder output
        :param n_upsampling_blocks: number of upsampling blocks
        :param n_residual_blocks: number of residual blocks
        :param s_dim: the length of the style code
        :param h_dim: the hidden dimension of the MLP
        """
        super(MUNITDecoder, self).__init__()
        c_after_upsampling = c_in // (2 ** n_upsampling_blocks)

        # Residual blocks with AdaIN
        self.munit_decoder_res_blocks = nn.ModuleList(
            [ResidualBlock(c_in, norm_type='AdaIN', s_dim=s_dim, h_dim=h_dim) for _ in range(n_residual_blocks)]
        )

        # Rest layers of MUNIT generator's decoder network
        self.munit_decoder_rest_layers = nn.Sequential(
            # Upsampling blocks
            *[self.get_upsampling_block(c_in // (2 ** i), c_in // 2 ** (i + 1)) for i in range(n_upsampling_blocks)],
            # Output layers
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(c_after_upsampling, 3, kernel_size=7)
            ),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Function for completing a forward pass of MUNITDecoder:
        Given an image tensor, representing the content, and a vector, representing the style, returns a new image
        tensor with the "style code" applied to the given "content code".
        :param x: image tensor of shape (N, C_in, H_c, W_c)
        :param s: vector tensor of shape (N, s_dim, 1, 1)
        :return: transformed image tensor of shape (N, 3(RGB) or 1(GS), H, W)
        """
        for res_block in self.munit_decoder_res_blocks:
            x = res_block(x, s)
        return self.munit_decoder_rest_layers(x)

    @staticmethod
    def get_upsampling_block(c_in: int, c_out: int, kernel_size: int = 5, padding: int = 2) -> nn.Sequential:
        """
        Get a block of the Upsampling part of the Decoder.
        :param c_in: number of channels of input tensor to block
        :param c_out: number of channels of output tensor from block
        :param kernel_size: kernel_size parameter of Conv2d layer
        :param padding: padding parameter of Conv2d layer
        :return: torch.nn.Sequential
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(padding=padding),
            nn.utils.spectral_norm(
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size)
            ),
            LayerNorm2d(c_out)
        )


class MUNITGenerator(nn.Module):
    """
    MUNITGenerator Class:
    This is the complete MUNIT autoencoder-type generator.
    """

    def __init__(self, c_base: int = 64, n_content_downsampling_blocks: int = 2, n_style_downsampling_blocks: int = 4,
                 n_residual_blocks: int = 4, s_dim: int = 8, h_dim: int = 256):
        """
        MUNITGenerator class constructor.
        :param c_base: number of channels in first convolutional layer
        :param n_content_downsampling_blocks: number of downsampling layers for ContentEncoder part
        :param n_style_downsampling_blocks: number of downsampling layers for StyleEncoder part
        :param n_residual_blocks: number of residual blocks
        :param s_dim:the dimension of the style tensor (s)
        :param h_dim:the hidden dimension of the MLP
        """
        super(MUNITGenerator, self).__init__()
        self.c_enc = MUNITContentEncoder(c_base=c_base, n_downsampling_blocks=n_content_downsampling_blocks,
                                         n_residual_blocks=n_residual_blocks)
        self.s_enc = MUNITStyleEncoder(c_base=c_base, n_downsampling_blocks=n_style_downsampling_blocks, s_dim=s_dim)
        self.dec = MUNITDecoder(self.c_enc.channels, n_upsampling_blocks=n_content_downsampling_blocks,
                                n_residual_blocks=n_residual_blocks, s_dim=s_dim, h_dim=h_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Complete a forward pass through the encoder part of the Autoencoder part of MUNITGenerator.
        :param x: input image tensor of shape (N, 3(RGB) or 1(GS), H, W)
        :return: tuple containing the content & style codes
        """
        return self.c_enc(x), self.s_enc(x)

    def decode(self, content: Tensor, style: Tensor) -> Tensor:
        """
        Complete a forward pass through the decoder part of the Autoencoder part of MUNITGenerator.
        :param content: content code tensor of shape (N, C_in, H_c, W_c)
        :param style: style code tensor of shape (N, s_dim, 1, 1)
        :return: torch.Tensor
        """
        # Reshape style (to be able to pass through AdaIN's MLP blocks): (N, s_dim, 1, 1) --> (N, s_dim)
        return self.dec(content, style.view(len(style), -1))

    ########################################
    # --------> Generator Losses <-------- #
    ########################################

    def ae_image_recon_loss(self, x: Tensor, criterion: nn.modules.Module = nn.L1Loss()) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """
        Autoencoder image reconstruction loss.
        :param x: input to encoders
        :param criterion: reconstruction criterion (defaults to L1)
        :return: tuple containing: 1) tensor with Autoencoder image reconstruction loss, 2) tensor with content codes
        and 3) tensor with style codes
        """
        c, s = self.encode(x)
        return criterion(x, self.decode(c, s)), c, s

    def ae_latent_recon_loss(self, c, s, criterion: nn.modules.Module = nn.L1Loss()) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Autoencoder image reconstruction loss. Given latent codes how well decoder + encoder can reconstruct those
        latent codes.
        :param c: content tensor
        :param s: style tensor
        :param criterion: reconstruction criterion (defaults to L1)
        :return: tuple containing: 1) tensor with loss for content code reconstruction, 2) for style code reconstruction
        and 3) reconstructed images from MUNITDecoder
        """
        x_hat = self.decode(c, s)
        recon = self.encode(x_hat)
        return criterion(recon[0], c), criterion(recon[1], s), x_hat
