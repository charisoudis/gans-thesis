from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional

from modules.partial.encoding import MLPBlock


class PixelNorm2d(nn.Module):
    """
    PixelNorm2d class:
    This is the per pixel normalization layer. This will divide each (x, y) pixel by channel's root mean square.
    """

    def __init__(self, eps: float = 1e-8):
        """
        PixelNorm2d class constructor.
        :param eps: added in formula's denominator to avoid division by zero
        """
        super(PixelNorm2d, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of PixelNorm2d:
        Given an image tensor, and for every spatial location (aka pixel) in this image it computes the
        sqrt(avg(pixel_channels .^ 2)).
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, 1, H, W)
        """
        x_norm = x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps) ** 0.5
        assert isinstance(x_norm, Tensor)
        return x_norm


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


class AdaptiveInstanceNorm2d(nn.Module):
    """
    AdaptiveInstanceNorm2d Class:
    This is a instance normalization with added learned mean & std for every channel (same in batch).
    Values:
        c_in: the number of channels the image has
        w_dim: the dimension of the style tensor (s)
        h_dim: the hidden dimension of the MLP
    """

    def __init__(self, c_in: int, w_dim: int = 8, h_dim: int = 256, n_mpl_blocks: int = 3):
        """
        AdaptiveInstanceNorm2d class constructor.
        :param (int) c_in: number of input channels
        :param (int) w_dim: length of style vector
        :param (int) h_dim: number of hidden neurons in MLPBlock's (that affinely transform instance norm statistics)
        :param (int) n_mpl_blocks: number of blocks in MLPBlock's (that affinely transform instance norm statistics)
        """
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(c_in, affine=False)
        self.style_scale_transform = MLPBlock(in_dim=w_dim, hidden_dim=h_dim, out_dim=c_in, n_blocks=n_mpl_blocks)
        self.style_shift_transform = MLPBlock(in_dim=w_dim, hidden_dim=h_dim, out_dim=c_in, n_blocks=n_mpl_blocks)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Function for completing a forward pass of AdaIN:
        Given an image and a style, returns the normalized image that has been scaled and shifted by the style.
        :param (torch.Tensor) x: the feature map of shape (N, C, H, W)
        :param (torch.Tensor) w: the intermediate noise vector w to be made into the style (y), of shape (NxW_dim)
        :return: torch.Tensor
        """
        normalized_x = self.instance_norm(x)  # (N,C,W,H)
        style_scale = self.style_scale_transform(w)[:, :, None, None]  # (N,C,1,1)
        style_shift = self.style_shift_transform(w)[:, :, None, None]  # (N,C,1,1)
        return style_scale * normalized_x + style_shift


class ModulatedConv2d(nn.Module):
    """
    ModulatedConv2d Class:
    This class implements a simpler version of the modulated convolution proposed in StyleGANv2 to fix artifacts created
    by AdaptiveInstanceNorm2d in the original StyleGAN.
    """

    def __init__(self, s_dim: int, c_in: int, c_out: int, kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 1):
        """
        ModulatedConv2d class constructor.
        :param s_dim: the dimension of the style vector, s
        :param c_in: the number of channels the expected input tensor should have
        :param c_out: the number of channels the expected output tensor should have
        :param kernel_size: kernel_size param of Conv2d layer
        :param padding: padding param of Conv2d layer
        """
        super(ModulatedConv2d, self).__init__()

        # Weights of the conv2d filters (each C_in x K_h x K_w, there will be C_out of them)
        # Weights are randomly initialized
        k_h = kernel_size[0] if type(kernel_size) == tuple else kernel_size
        k_w = kernel_size[1] if type(kernel_size) == tuple else kernel_size
        self.conv_weight = nn.Parameter(torch.randn(1, c_out, c_in, k_h, k_w))
        # FC layer ([N,w_dim] --> [N,C_in])
        self.style_scale_transform = nn.Linear(s_dim, c_in)
        self.eps = 1e-6
        # Save for forward pass
        self.c_out = c_out
        self.padding = padding
        self.stride = stride

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Forward pass of layer.
        :param x: input "image" tensor of shape (N, C_in, H, W)
        :param s: input style tensor of shape (N, w_dim)
        :return: a torch.Tensor object of shape (N, C_out, H, W)
        """
        batch_size, c_in, h, w = x.shape

        # Calculate s_i, for the i-th channel (i in range [0, self.c_in])
        style_scale = self.style_scale_transform(s)
        style_scale = style_scale.view(batch_size, 1, c_in, 1, 1)
        #  W_dot_ijk = s_i * W_ijk
        w_dot = self.conv_weight * style_scale
        # W_dot.shape = (N, C_out, C_in, K_h, K_w)

        # W_dot_dot_ijk = W_dot_ijk / (var(j) + eps)
        w_dot_dot = w_dot / torch.sqrt(
            (w_dot ** 2).sum([2, 3, 4])[:, :, None, None, None] + self.eps
        )
        # W_dot_dot.shape = (N, C_out, C_in, K_h, K_w)

        # Now, the trick is that we'll make the images into one image, and
        # all of the conv filters into one filter, and then use the "groups"
        # parameter of F.conv2d to apply them all at once
        x = x.view(1, batch_size * c_in, h, w)
        # x.shape = (1, N * C_in, H, W)
        kernel = w_dot_dot.view(batch_size * self.c_out, c_in, *w_dot_dot.shape[3:])
        # kernel.shape = (N * C_out, C_in, H, W)
        out = functional.conv2d(x, kernel, stride=self.stride, padding=self.padding, groups=batch_size)
        # out.shape = (1, N * C_out, C_in, H, W)
        return out.view(batch_size, out.shape[1] // batch_size, *out.shape[2:])


class BatchStd(nn.Module):
    """
    MiniBatchStd Class:
    Add mini-batch std (as described in ProGAN) as the last channel of disc to improve variance.
    """

    def __init__(self, group_size: int = 4):
        """
        MiniBatchStd class constructor.
        :param (int) group_size: size of each group (batch is split into M such groups)
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass MiniBatchStd class: computes the mini-batch str (M=N/G values) and appends it as the last channel.
        :param (torch.Tensor) x: the input tensor
        :return: the input tensor + 1 channel containing the mini-batch std
        """
        shape = x.shape  # (N,C,H,W)
        # (M,G,C,H,W): split batch into M groups of size |G|
        try:
            x_std = x.view(self.group_size, -1, shape[1], shape[2], shape[3])
        except RuntimeError:
            x = x[0:shape[0] - (shape[0] % (self.group_size * shape[1] * shape[2] * shape[3]))]
            x_std = x.view(self.group_size, -1, shape[1], shape[2], shape[3])
        # (M,G,C,H,W): remove mean across groups
        x_std -= torch.mean(x_std, dim=0, keepdim=True)
        # (M,C,H,W): compute std across groups
        x_std = (torch.mean(x_std ** 2, dim=0, keepdim=False) + 1e-08) ** 0.5
        # (M,1,1,1): Take average across CHW
        x_std = torch.mean(x_std.view(int(shape[0] / self.group_size), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        # (N,1,H,W): Expand to same shape as x with one channel (by repeating the M values)
        x_std = x_std.repeat(self.group_size, 1, shape[2], shape[3])
        # Append as new channel
        return torch.cat([x, x_std], dim=1)

    def __repr__(self):
        return self.__class__.__name__ + f'(Group Size = {self.group_size})'


if __name__ == '__main__':
    # ml = ModulatedConv2d(32, 4, 8, kernel_size=3, stride=3, padding=1)
    # _x = torch.randn(1, 4, 128, 128)
    # _s = torch.randn(1, 32)
    # _out = ml(_x, _s)
    # print(_out.shape)

    # _x = torch.randn(4, 128, 4, 4)
    # _x_dot = BatchStd()(_x)
    # print(_x_dot[:, -1, 2, 0])
    pass
