import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ExpandingBlock(nn.Module):
    """
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to up-sample, with an optional instance norm
    Values:
        c_in: the number of channels to expect from a given input
    """

    STATE_DICT_REPLACE_DICT = {
        '.expanding_block.0.': '.upscale.',
        '.expanding_block.1.': '.expanding_block.0.',
    }

    @staticmethod
    def fix_state_dict(state_dict: dict) -> dict:
        """
        Fix state dicts after recent update in layer naming
        :param (dict) state_dict: old state dict
        :return: a dict object with the updated keys
        """
        # Check if newer checkpoint
        for key in list(state_dict.keys()):
            if '.upscale.' in key:
                return state_dict
        # Fix older checkpoints
        s_r_dict = ExpandingBlock.STATE_DICT_REPLACE_DICT
        for key in list(state_dict.keys()):
            for search_key in list(s_r_dict.keys()):
                if search_key in key:
                    state_dict[key.replace(search_key, s_r_dict[search_key])] = state_dict[key]
                    del state_dict[key]
        return state_dict

    def __init__(self, c_in: int, use_norm: bool = True, kernel_size: int = 3, activation: Optional[str] = 'relu',
                 output_padding: int = 1, stride: int = 2, padding: int = 1, c_out: Optional[int] = None,
                 norm_type: str = 'instance', use_dropout: bool = False, use_skip: bool = False):
        """
        ExpandingBlock class constructor.
        :param (int) c_in: number of input channels
        :param (bool) use_norm: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param (int) kernel_size: filter (kernel) size of torch.nn.Conv2d
        :param (str) activation: type of activation function used (supported: 'relu', 'lrelu')
        :param (int) output_padding: output_padding of torch.nn.ConvTranspose2d
        :param (int) stride: stride of torch.nn.ConvTranspose2d (defaults to 2 for our needs)
        :param (int) padding: padding of torch.nn.ConvTranspose2d
        :param (str) c_out: number of output channels
        :param (str) norm_type: available types are 'batch', 'instance', 'pixel', 'layer'
        :param (bool) use_dropout: set to True to add a `nn.Dropout()` layer with probability of 0.2
        :param (bool) use_skip: set to True to enable UNET-like behaviour
        """
        super(ExpandingBlock, self).__init__()
        c_out = c_in // 2 if c_out is None else c_out

        # Upscaling layer using transposed convolution
        # noinspection PyTypeChecker
        self.upscale = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                          output_padding=output_padding)
        _layers = []
        if use_skip:
            # noinspection PyTypeChecker
            _layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))

        if use_norm:
            from modules.partial.normalization import PixelNorm2d, LayerNorm2d
            switcher = {
                'batch': nn.BatchNorm2d(c_out),
                'instance': nn.InstanceNorm2d(c_out),
                'pixel': PixelNorm2d(),
                'layer': LayerNorm2d(c_out),
            }
            _layers.append(switcher[norm_type])
        if use_dropout:
            _layers.append(nn.Dropout2d(p=0.2))
        if activation is not None:
            activations_switcher = {
                'relu': nn.ReLU(),
                'lrelu': nn.LeakyReLU(0.2),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
            }
            _layers.append(activations_switcher[activation])
        self.expanding_block = nn.Sequential(*_layers)

        self.c_out = c_out
        self.use_skip = use_skip

    def forward(self, x: Tensor, skip_conn_at_x: Optional[Tensor] = None) -> Tensor:
        """
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        :param (Tensor) x: image tensor of shape (N, C, H, W)
        :param (Tensor) skip_conn_at_x: the image tensor from the contracting path (from the opposing block of x)
                                        for the skip connection
        :return: transformed image tensor of shape (N, C/2, H*2, W*2)
        """
        # Check if skip connection is present
        assert self.use_skip is False or skip_conn_at_x is not None, 'use_skip was set, but skip_conn_at_x is None!'
        # Upscale current input
        x = self.upscale(x)
        # Append skip connection (if one exists)
        if self.use_skip:
            # Specify cat()'s dim to be 1 (aka channels), since we want a channel-wise concatenation of the two tensors
            x = torch.cat([x, UNETExpandingBlock.crop_skip_connection(skip_conn_at_x, x.shape)], dim=1)
        return self.expanding_block(x)


class UNETExpandingBlock(nn.Module):
    """
    UNETExpandingBlock Class:
    Symmetric to the UNETContracting block, with the addition of skip connection which concatenates the output of the
    respective contracting block.
    Attention: Unlike UNET paper, we add padding=1 to Conv2d layers to make a "symmetric" version of UNET.
    """

    def __init__(self, c_in: int, use_bn: bool = True, use_dropout: bool = False, activation: str = 'relu'):
        """
        UNETExpandingBlock class constructor.
        :param c_in: number of input channels
        :param use_bn: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param use_dropout: indicates if Dropout is applied after BatchNorm2d
        :param activation: type of activation function used (supported: 'relu', 'lrelu')
        """
        super(UNETExpandingBlock, self).__init__()
        # Up-sample + convolution (instead of transposed convolution)
        # noinspection PyTypeChecker
        self.upsample_and_1st_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c_in, c_in // 2, kernel_size=2)
        )
        # Then, the output is concatenated with encoder's output at same level
        # Then, the rest of the ExpandingBlock architecture follows
        # noinspection PyTypeChecker
        self._2nd_and_3rd_conv = nn.Sequential(
            # 2nd convolution layer
            nn.Conv2d(c_in, c_in // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_in // 2) if use_bn else nn.Identity(),
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2),
            # 3rd convolution layer
            nn.Conv2d(c_in // 2, c_in // 2, kernel_size=2, padding=1),
            nn.BatchNorm2d(c_in // 2) if use_bn else nn.Identity(),
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor, skip_conn_at_x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        :param (Tensor) x: image tensor of shape (N, C, H, W)
        :param (Tensor) skip_conn_at_x: the image tensor from the contracting path (from the opposing block of x)
                                        for the skip connection
        :return: the transformed image tensor of shape (N, C/2, H*2, W*2)
        """
        x = self.upsample_and_1st_conv(x)
        # Specify cat()'s dim to be 1 (aka channels), since we want a channel-wise concatenation of the two tensors
        x = torch.cat([x, UNETExpandingBlock.crop_skip_connection(skip_conn_at_x, x.shape)], dim=1)
        return self._2nd_and_3rd_conv(x)

    @staticmethod
    def crop_skip_connection(skip_con: Tensor, shape: torch.Size) -> Tensor:
        """
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels. Crops (H, W) dims of input tensor.
        :param skip_con: image tensor of shape (N, C, H, W)
        :param shape: a torch.Size object with the shape you want skip_con to have: (N, C, H_hat, W_hat)
        :return: torch.Tensor of shape (N, C, H_hat, W_hat)
        """
        new_w = shape[-1]
        start_index = math.ceil((skip_con.shape[-1] - new_w) / 2.0)
        cropped_skip_con = skip_con[:, :, start_index:start_index + new_w, start_index:start_index + new_w]
        return cropped_skip_con


class FeatureMapLayer(nn.Module):
    """
    FeatureMapLayer Class:
    The final layer of a Generator; maps each the output to the desired number of output channels
    Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
    """

    def __init__(self, c_in: int, c_out: int):
        """
        FeatureMapLayer class constructor.
        :param c_in: number of output channels
        """
        super(FeatureMapLayer, self).__init__()
        # noinspection PyTypeChecker
        self.feature_map_block = nn.Conv2d(c_in, c_out, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of FeatureMapLayer:
        Given an image tensor, returns it mapped to the desired number of channels.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        return self.feature_map_block(x)


class ChannelsProjectLayer(nn.Module):
    """
    ChannelsProjectLayer Class:
    Layer to project C_in channels of input tensor to C_out channels in output tensor
    Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
    """

    def __init__(self, c_in: int, c_out: int, use_spectral_norm: bool = False, padding: int = 0):
        """
        ChannelsProjectLayer class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_out: number of output channels
        :param (bool) use_spectral_norm: set to True to add a spectral normalization layer after the Conv2d
        :param (int) padding: nn.Conv2d's padding argument
        """
        super(ChannelsProjectLayer, self).__init__()
        # noinspection PyTypeChecker
        self.feature_map_block = nn.Conv2d(c_in, c_out, stride=1, kernel_size=1, padding=padding)
        if use_spectral_norm:
            self.feature_map_block = nn.utils.spectral_norm(self.feature_map_block)

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of ChannelsProjectLayer:
        Given an image tensor, returns it mapped to the desired number of channels.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        return self.feature_map_block(x)
