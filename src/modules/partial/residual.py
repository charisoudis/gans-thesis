import torch.nn as nn
from torch import Tensor

from modules.partial.normalization import AdaptiveInstanceNorm2d


class ResidualBlock(nn.Module):
    """
    ResidualBlock Class:
    Given an input torch.Tensor of shape (N, C_in, H_in, W_in) it outputs a torch.Tensor of shape
    (N, C_out, H_out, W_out), where almost everywhere (C,H,W)_in = (C,H,W)_out.
    """

    def __init__(self, c_in: int, norm_type: str = 'IN', s_dim: int = None, h_dim: int = None):
        """
        ResidualBlock class constructor.
        :param c_in: number of input channels
        :param norm_type: type of normalization layer used after Conv layers (supported: 'AdaIN', 'IN', 'BN')
        :param s_dim: (when norm_type=='AdaIN'): length of style vector
        :param h_dim: (when norm_type=='AdaIN'): number of neurons in hidden layers of AdaIN's transformation MLPs
        """
        super(ResidualBlock, self).__init__()
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, padding_mode='reflect')
        # Use InstanceNorm2d instead of BatchNorm2d because different images in batch may contain
        # very different styles and thus BatchNorm2 will average those tending to create blurry results
        self.norm1 = nn.BatchNorm2d(c_in) if norm_type == 'BN' else (
            nn.InstanceNorm2d(c_in) if norm_type == 'IN' else AdaptiveInstanceNorm2d(c_in, s_dim, h_dim)
        )
        self.activation = nn.ReLU()
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm2 = nn.BatchNorm2d(c_in) if norm_type == 'BN' else (
            nn.InstanceNorm2d(c_in) if norm_type == 'IN' else AdaptiveInstanceNorm2d(c_in, s_dim, h_dim)
        )
        self.use_style = norm_type == 'AdaIN'

    def forward(self, x: Tensor, s: Tensor = None) -> Tensor:
        """
        Function for completing a forward pass of ResidualBlock:
        Given an image tensor, completes a residual block and returns the transformed tensor.
        :param x: image tensor of shape (N, C, H, W)
        :param s: style vector of shape (N, s_dim)
        :return: transformed image tensor
        """
        x_initial = x
        x = self.conv1(x)
        x = self.norm1(x, w=s) if self.use_style else self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x, w=s) if self.use_style else self.norm2(x)
        return x + x_initial
