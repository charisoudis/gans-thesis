import torch.nn as nn
from torch import Tensor


class ExpandingBlock(nn.Module):
    """
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to up-sample, with an optional instance norm
    Values:
        c_in: the number of channels to expect from a given input
    """

    def __init__(self, c_in: int, use_bn: bool = True, kernel_size: int = 3, activation: str = 'relu'):
        """
        ExpandingBlock class constructor.
        :param c_in: number of input channels
        :param use_bn: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param kernel_size: filter (kernel) size
        :param activation: type of activation function used (supported: 'relu', 'lrelu')
        """
        super(ExpandingBlock, self).__init__()
        self.expanding_block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(c_in // 2) if use_bn else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        :param x: image tensor of shape (N, C, H, W)
        :return: transformed image tensor of shape (N, C/2, H*2, W*2)
        """
        return self.expanding_block(x)


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

    def __init__(self, c_in: int, c_out: int, use_spectral_norm: bool = False):
        """
        ChannelsProjectLayer class constructor.
        :param c_in: number of output channels
        """
        super(ChannelsProjectLayer, self).__init__()
        self.feature_map_block = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0)
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
