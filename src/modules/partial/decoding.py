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

# class ExpandingBlock(nn.Module):
#     '''
#     ExpandingBlock Class:
#     Performs an upsampling, a convolution, a concatenation of its two inputs,
#     followed by two more convolutions with optional dropout
#     Values:
#         input_channels: the number of channels to expect from a given input
#     '''
#     def __init__(self, input_channels, use_dropout=False, use_bn=True):
#         super(ExpandingBlock, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
#         self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
#         if use_bn:
#             self.batchnorm = nn.BatchNorm2d(input_channels // 2)
#         self.use_bn = use_bn
#         self.activation = nn.ReLU()
#         if use_dropout:
#             self.dropout = nn.Dropout()
#         self.use_dropout = use_dropout
#
#     def forward(self, x, skip_con_x):
#         '''
#         Function for completing a forward pass of ExpandingBlock:
#         Given an image tensor, completes an expanding block and returns the transformed tensor.
#         Parameters:
#             x: image tensor of shape (batch size, channels, height, width)
#             skip_con_x: the image tensor from the contracting path (from the opposing block of x)
#                     for the skip connection
#         '''
#         x = self.upsample(x)
#         x = self.conv1(x)
#         skip_con_x = crop(skip_con_x, x.shape)
#         x = torch.cat([x, skip_con_x], axis=1)
#         x = self.conv2(x)
#         if self.use_bn:
#             x = self.batchnorm(x)
#         if self.use_dropout:
#             x = self.dropout(x)
#         x = self.activation(x)
#         x = self.conv3(x)
#         if self.use_bn:
#             x = self.batchnorm(x)
#         if self.use_dropout:
#             x = self.dropout(x)
#         x = self.activation(x)
#         return x
class UNETExpandingBlock(nn.Module):
    """
    UNETExpandingBlock Class:
    Performs a convolutional transpose operation in order to up-sample, with an optional instance norm
    Values:
        c_in: the number of channels to expect from a given input
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
        self.expanding_block = nn.Sequential(
            # Up-sample before convolutions (instead of transposed convolutions)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 1st convolution layer
            nn.Conv2d(c_in, c_in // 2, kernel_size=2),
            nn.BatchNorm2d(c_in * 2) if use_bn else nn.Identity(),
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2),
        )
        # TODO: finish this class

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
