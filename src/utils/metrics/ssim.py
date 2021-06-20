import math
from math import exp
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch.autograd import Variable
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader

from datasets.deep_fashion import ICRBCrossPoseDataset, ICRBDataset
from modules.generators.pgpg import PGPGGenerator
from utils.dep_free import get_tqdm
from utils.filesystems.local import LocalFolder, LocalCapsule
from utils.ifaces import Freezable


def _ssim_map(img1: Tensor, img2: Tensor, window, window_size: int, c_img: int) -> Tensor:
    """
    Function to calculate the SSIM difference maps between two (batches of) images.
    :param img1: the 1st image batch (real image or perfect reconstruction)
    :param img2: the 2nd image batch (real image or perfect reconstruction)
    :param window: the convolution kernel (should be in the same device as image tensors)
    :param window_size: width (or height, is the same) of window kernel
    :param c_img: the image channels of both img1 and img2 batches
    :return: a torch.Tensor object with SSIM difference maps
    """
    padding = window_size // 2
    mu1 = functional.conv2d(img1, window, padding=padding, groups=c_img)
    mu2 = functional.conv2d(img2, window, padding=padding, groups=c_img)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = functional.conv2d(img1 * img1, window, padding=padding, groups=c_img) - mu1_sq
    sigma2_sq = functional.conv2d(img2 * img2, window, padding=padding, groups=c_img) - mu2_sq
    sigma12 = functional.conv2d(img1 * img2, window, padding=padding, groups=c_img) - mu1_mu2

    c_1 = 0.01 ** 2
    c_2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c_1) * (2 * sigma12 + c_2)) / ((mu1_sq + mu2_sq + c_1) * (sigma1_sq + sigma2_sq + c_2))
    return ssim_map


def _ssim(img1: Tensor, img2: Tensor, window, window_size: int, c_img: int, size_average=True) -> Tensor:
    """
    Function to calculate the SSIM index between the two given batches of images, :attr:`img1` and :attr:`img2`.
    :param img1: the 1st image batch (real image or perfect reconstruction)
    :param img2: the 2nd image batch (real image or perfect reconstruction)
    :param window: the convolution kernel (should be in the same device as image tensors)
    :param window_size: width (or height, is the same) of window kernel
    :param c_img: the image channels of both img1 and img2 batches
    :param size_average: set to True to average SSIM index over all image channels, else :attr:`c_img` SSIM indices will
                         be returned, one for each channel
    :return: a torch.Tensor object with the overall mean SSIM index if :attr:`size_average`=True or with the SSIM
             indices for each channel if False
    """
    ssim_map = _ssim_map(img1, img2, window=window, window_size=window_size, c_img=c_img)
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    """
    SSIM Class:
    This class is used to compute the Structural Similarity (SSIM) index between two batches of images.
    Source: https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, n_samples: int = 512, batch_size: int = 8, device: str = 'cpu', c_img: int = 3,
                 window_size: int = 11, size_average: bool = True):
        """
        SSIM class constructor:
        :param n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                          number gets, the more accurate the metric is)
        :param batch_size: the number of samples to precess at each loop
        :param device: device to run computation on (defaults to 'cpu')
        :param c_img: number of image channels (defaults to 3 for RGB images)
        :param window_size: SSIM window size parameter (kernel size of Conv2d filters)
        :param size_average: SSIM size average flag (set to True to output a scalar value of SSIM index)
        """
        super(SSIM, self).__init__()
        self.tqdm = get_tqdm()

        # Create convolution kernel (a multivariate gaussian)
        self.window = SSIM._create_window(window_size, c_img)
        self.window = self.window.to(device)

        # Save arguments to instance
        self.window_size = window_size
        self.c_img = c_img
        self.size_average = size_average
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> Tensor:
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @staticmethod
    def _create_window(window_size: int, c_img: int, gaussian_sigma: float = 1.5) -> Variable:
        _1D_window = SSIM._gaussian(window_size, gaussian_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return Variable(_2D_window.expand(c_img, 1, window_size, window_size).contiguous())

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                show_progress: bool = True, skip_asserts: bool = False, **kwargs) -> float:
        """
        Compute the Inception Score of the images generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images as inputs to the Generator
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param skip_asserts: set to True to skip normalization checks (e.g. when the generator has not been trained and
                             outputs close-to-zero noise)
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :return: a scalar value with the computed IS
        """
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        if self.device == 'cuda:0' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        assert isinstance(gen, Freezable), 'Generator should implement utils.ifaces.Freezable'
        with gen.frozen():
            cur_samples = 0
            ssim_maps_list = []
            break_after = False
            for real_samples in self.tqdm(dataloader, total=int(math.ceil(self.n_samples / self.batch_size)),
                                          disable=not show_progress, desc="SSIM"):
                if cur_samples >= self.n_samples:
                    break_after = True

                # Get target (real) images
                target_output = real_samples[target_index] if target_index is not None else real_samples
                target_output = target_output.to(self.device)
                # if not skip_asserts:
                #     assert target_output.min() < 0, f'target_output.min() < 0: FAILED, min={target_output.min()}'
                #     assert target_output.min() >= -1, f'target_output.min() >= -1: FAILED, min={target_output.min()}'
                #     assert target_output.max() > 0, f'target_output.max() > 0: FAILED, max={target_output.max()}'
                #     assert target_output.max() <= 1, f'target_output.max() <= 1: FAILED, max={target_output.max()}'

                cur_batch_size = len(target_output)

                # Generate fake images from conditions
                gen_inputs = [real_samples[_i].to(self.device) for _i in condition_indices] if condition_indices \
                    else torch.randn(cur_batch_size, z_dim, device=self.device)
                fake_output = gen(*gen_inputs if type(gen_inputs) == list else gen_inputs)
                if type(fake_output) == tuple or type(fake_output) == list:
                    fake_output = fake_output[-1]
                # if not skip_asserts:
                #     assert fake_output.min() < 0, f'fake_output.min() < 0: FAILED, min={fake_output.min()}'
                #     assert fake_output.min() >= -1, f'fake_output.min() >= -1: FAILED, min={fake_output.min()}'
                #     assert fake_output.max() > 0, f'fake_output.max() > 0: FAILED, max={fake_output.max()}'
                #     assert fake_output.max() <= 1, f'fake_output.max() <= 1: FAILED, max={fake_output.max()}'

                # Compute SSIM difference maps
                ssim_maps_list.append(_ssim_map(target_output, fake_output, self.window, self.window_size, self.c_img))
                cur_samples += cur_batch_size

                if break_after:
                    break

            # Compute SSIM from difference maps
            ssim_maps = torch.cat(ssim_maps_list, dim=0).cpu()

        return ssim_maps.mean().float() if self.size_average else ssim_maps.mean(1).mean(1).mean(1).float()


# noinspection DuplicatedCode
if __name__ == '__main__':
    # Init Google Drive stuff
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _groot = LocalFolder.root(LocalCapsule(_local_gdrive_root))
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Setup evaluation dataset
    _target_shape = 128
    _target_channels = 3
    _dataset = ICRBCrossPoseDataset(dataset_fs_folder_or_root=_datasets_groot,
                                    image_transforms=ICRBDataset.get_image_transforms(_target_shape, _target_channels))

    # Initialize Generator
    _gen = PGPGGenerator(c_in=2 * _target_channels, c_out=_target_channels, w_in=_target_shape, h_in=_target_shape)

    # Evaluate Generator using FID
    _ssim_calculator = SSIM(n_samples=2, batch_size=1, device='cpu')
    _ssim = _ssim_calculator(_dataset, _gen, target_index=1, condition_indices=(0, 2), show_progress=True)
    print(_ssim)
