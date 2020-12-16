import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.encoding import MLPBlock


class PixelNormalizationLayer(nn.Module):
    """
    PixelNormalizationLayer class:
    This is the per pixel normalization layer. This will divide each (x, y) pixel by channel's root mean square.
    """

    def __init__(self, eps: float = 1e-8):
        """
        PixelNormalizationLayer class constructor.
        :param eps: added in formula's denominator to avoid division by zero
        """
        super(PixelNormalizationLayer, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of PixelNormalizationLayer:
        Given an image tensor, and for every spatial location (aka pixel) in this image it computes the
        sqrt(avg(pixel_channels .^ 2)).
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, 1, H, W)
        """
        x_norm = x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5
        assert isinstance(x_norm, Tensor)
        return x_norm


class AdaptiveInstanceNorm2d(nn.Module):
    """
    AdaptiveInstanceNorm2d Class:
    This is a instance normalization with added learned mean & std for every channel (same in batch).
    Values:
        c_in: the number of channels the image has
        s_dim: the dimension of the style tensor (s)
        h_dim: the hidden dimension of the MLP
    """

    def __init__(self, c_in: int, s_dim: int = 8, h_dim: int = 256):
        """
        AdaptiveInstanceNorm2d class constructor.
        :param c_in: number of input channels
        :param s_dim: length of style vector
        :param h_dim: number of hidden neurons in MLPBlock modules that affinely transform instance norm statistics
        """
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(c_in, affine=False)
        self.style_scale_transform = MLPBlock(in_dim=s_dim, hidden_dim=h_dim, out_dim=c_in)
        self.style_shift_transform = MLPBlock(in_dim=s_dim, hidden_dim=h_dim, out_dim=c_in)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Function for completing a forward pass of AdaIN:
        Given an image and a style, returns the normalized image that has been scaled and shifted by the style.
        :param x: the feature map of shape (N, C, H, W)
        :param w: the intermediate noise vector w to be made into the style (y)
        :return: torch.Tensor
        """
        normalized_x = self.instance_norm(x)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        return style_scale * normalized_x + style_shift


class LayerNorm2d(nn.Module):
    """
    LayerNorm2d Class:
    Performs Layer-wise normalization according to Hinton's paper. LayerNorm2d is placed as follows:
        - InstanceNorm2d: per instance AND per channel
        - BatchNorm2d: per channel
        - LayerNorm2d: per instance
    """

    def __init__(self, c_in: int, eps: float = 1e-5, affine: bool = True):
        """
        LayerNorm2d class constructor.
        :param c_in: number of channels in input tensor
        :param eps: epsilon parameter to avoid division by zero
        :param affine: whether to apply affine de-normalization
        """
        super(LayerNorm2d, self).__init__()
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.rand(c_in))
            self.beta = nn.Parameter(torch.zeros(c_in))

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of LayerNorm2d:
        Given an image tensor it returns the normalized image that has been scaled and shifted by the affine transform.
        :param x: the feature map of shape (N, C, H, W)
        :return: torch.Tensor
        """
        mean = x.flatten(1).mean(1).reshape(-1, 1, 1, 1)
        std = x.flatten(1).std(1).reshape(-1, 1, 1, 1)
        x = (x - mean) / (std + self.eps)
        return x * self.gamma.reshape(1, -1, 1, 1) + self.beta.reshape(1, -1, 1, 1) if self.affine else x
