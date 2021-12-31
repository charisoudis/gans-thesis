# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Perceptual Path Length (PPL) from the paper "A Style-Based Generator Architecture for Generative Adversarial Networks".
Matches the original implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
"""

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from modules.classifiers.vgg16 import VGG16Sliced
from utils.dep_free import get_tqdm
from utils.ifaces import FilesystemFolder
from utils.pytorch import ToTensorOrPass


def slerp(a: torch.Tensor, b: torch.Tensor, t) -> torch.Tensor:
    """
    Spherical interpolation of a batch of vectors.
    :param (torch.Tensor) a:
    :param (torch.Tensor) b:
    :param t:
    :return: a torch.Tensor object containing len(t) spherical interpolations between a and b.
    """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


# ----------------------------------------------------------------------------

class PPLSampler(nn.Module):
    """
    PPLSampler Class:
    Source: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/perceptual_path_length.py
    """

    # These are the VGG16 image transforms
    VGG16Transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToTensorOrPass(renormalize=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Keep the Inception network in a static variable to avoid re-initializing it on sub-classes
    VGG16Sliced = None

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, device: str, n_samples: int, batch_size: int = 8,
                 epsilon: float = 1e-4, space: str = 'w', sampling: str = False, crop: bool = False):
        """
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` instance for cloud or
                                                           locally saved models using the same API
        :param (str) device: the device type on which to run the Inception model (defaults to 'cpu')
        :param (int) n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                                number gets, the more accurate the metric is)
        :param (float) epsilon: step dt
        :param (str) space: one of 'z', 'w'
        :param (str) sampling: one of 'full', 'end'
        :param (bool) crop:
        """
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super(PPLSampler, self).__init__()
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop

        # Instantiate Inception v3 model(s)
        if PPLSampler.VGG16Sliced is None:
            _vgg16_instance = VGG16Sliced(model_fs_folder_or_root, chkpt_step='397923af')
            PPLSampler.VGG16Sliced = _vgg16_instance.to(device).eval()

        # Save params in instance
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.tqdm = get_tqdm()

        # Disable grad graphs in PPL Sampler
        self.eval().requires_grad_(False).to(device)

    def sampler(self, c: torch.Tensor):
        # Generate random latents and interpolation t-values.
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)

        # Interpolate in W or Z.
        # TODO: This version only works on unlabelled data. If labels exist, they are ignored, whereas in StyleGAN2 they
        # TODO: are also to disentangle the noise space.
        if self.space == 'w':
            w0, w1 = self.G.noise_mapping(torch.cat([z0, z1])).chunk(2)
            wt0 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2))
            wt1 = w0.lerp(w1, t.unsqueeze(1).unsqueeze(2) + self.epsilon)
        else:  # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0, wt1 = self.G.noise_mapping(torch.cat([zt0, zt1])).chunk(2)

        # Generate images.
        img_t0 = self.G(w=wt0)
        img_t1 = self.G(w=wt1)

        # TODO: port this code into vanilla PyTorch
        # Center crop.
        # if self.crop:
        #     assert img.shape[2] == img.shape[3]
        #     c = img.shape[2] // 8
        #     img = img[:, :, c * 3: c * 7, c * 2: c * 6]
        #
        # # Downsample to 256x256.
        # factor = self.G.img_resolution // 256
        # if factor != 1:
        #     img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]) \
        #         .mean([3, 5])
        #
        # # Scale dynamic range from [-1,1] to [0,255].
        # img = (img + 1) * (255 / 2)
        # if self.G.img_channels == 1:
        #     img = img.repeat([1, 3, 1, 1])
        img_t0 = PPLSampler.VGG16Transforms(img_t0)
        img_t1 = PPLSampler.VGG16Transforms(img_t1)

        # Evaluate differential LPIPS
        lpips_t0 = PPLSampler.VGG16Sliced(img_t0)
        lpips_t1 = PPLSampler.VGG16Sliced(img_t1)
        return (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2

    # noinspection PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                show_progress: bool = True, **kwargs) -> Tensor:
        """
        Compute the Perception Path Length between random $self.n_samples$ images from the given dataset and equal
        number of images generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images. Attention: no transforms should be
                        applied when __getitem__ is called since the transforms are different on Inception v3
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :return: a scalar torch.Tensor object containing the computed PPL value
        """
        # Create the dataloader instance
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        if self.device == 'cuda:0' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Sampling loop.
        dist = []
        cur_samples = 0
        for real_samples in self.tqdm(dataloader, total=int(math.ceil(self.n_samples / self.batch_size)),
                                      disable=not show_progress, desc='PPL'):
            if cur_samples >= self.n_samples:
                break_after = True

            if condition_indices is None:
                # No labels --> Replace with zeros
                labels = torch.zeros((real_samples.shape[0], 0), dtype=np.float32)
            else:
                # Pass labels to sampler
                labels = real_samples[condition_indices[-1]].pin_memory().to(self.device)

            x = self.sampler(labels)
            dist.append(x)

        # Compute PPL
        dist = torch.cat(dist)[:cur_samples].cpu().numpy()
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return torch.tensor(ppl)
