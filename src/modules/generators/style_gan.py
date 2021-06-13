import torch
import torch.nn as nn
from torch import Tensor


class NoiseMappingLayer(nn.Module):
    """
    NoiseMappingLayer Class:
    This class implements a layer of the Noise Mapping network proposed in the original StyleGAN paper.
    """

    def __init__(self, z_dim: int, hidden_dim: int, w_dim: int):
        """
        NoiseMappingLayer class constructor.
        :param z_dim: the dimension of the noise vector
        :param hidden_dim: the inner dimension
        :param w_dim: the dimension of the w-vector (i.e. the vector in the hypothetically disentangled vector space)
        """
        super(NoiseMappingLayer, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Function for completing a forward pass of MappingLayers:
        Given an initial noise tensor, returns the intermediate noise tensor.
        :param z: a noise tensor with dimensions (N, z_dim)
        """
        return self.mapping(z)


class InjectNoise(nn.Module):
    """
    InjectNoise Class:
    This class implements the random noise injection that occurs before every AdaIN block of the original StyleGAN or
    before every ModulatedConv2d layer of StyleGANv2.
    """

    def __init__(self, c_in: int):
        """
        InjectNoise class constructor.
        :param c_in: the number of channels of the expected input tensor
        """
        super().__init__()
        # Initiate the weights for the channels from a random normal distribution
        # You use nn.Parameter so that these weights can be optimized
        self.weight = nn.Parameter(torch.randn(c_in).view(1, c_in, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of InjectNoise: Given an image, returns the image plus random noise.
        :param x: the input feature map of shape (N, C_in, H, W)
        :return: a torch.Tensor object of the same shape as $x$
        """
        # Noise :   (N,1,W,H)
        # Weight:   (N,C,1,1)
        batch_size, c_in, h, w = x.shape
        noise_shape = (batch_size, 1, h, w)
        noise = torch.randn(noise_shape, device=x.device)
        return x + self.weight * noise
