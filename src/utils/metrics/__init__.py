__all__ = ['fid', 'f1', 'ssim', 'is_', 'GanEvaluator']

from typing import Optional, Union, Dict

import torch
from torch import nn
# noinspection PyProtectedMember
from torch.utils.data import Dataset, Subset

from utils.ifaces import FilesystemFolder
from utils.metrics.f1 import F1
from utils.metrics.fid import FID
from utils.metrics.is_ import IS
from utils.metrics.ssim import SSIM


class GanEvaluator(object):
    """
    GanEvaluator Class:
    This class is used to evaluate the image generation performance of a GAN.
    """

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, gen_dataset: Dataset, n_samples: int = 1e4,
                 batch_size: int = 32, ssim_c_img: int = 3, device: torch.device or str = 'cpu',
                 target_index: Optional[int] = None, condition_indices: Optional[Union[int, tuple]] = None,
                 z_dim: Optional[int] = None, f1_k: int = 3):
        """
        GanEvaluator class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: absolute path to model checkpoints directory or
                                                           FilesystemFolder instance for cloud-synced models
        :param (Dataset) gen_dataset: the dataset used to train the generator as a `torch.utils.data.Dataset` object
        :param (int) n_samples: the number of samples (images) used to evaluate GAN performance (the bigger the number
                                of samples, the more robust the evaluation metric)
        :param (int) batch_size: the batch size to access the dataset's images
        :param (int) ssim_c_img: the number of images' channels (3 for RGB, 1 for grayscale); needed by the SSIM metric
        :param (str) device: the device on which to run the calculations (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (int or None) target_index: index of target (real) output from the arguments that returns
                                           dataset::__getitem__() method
        :param (int) z_dim: if :attr:`condition_indices` is `None`, then this is necessary to produce random noise to
                            feed into the DCGAN-like generator
        :param (int or tuple or None) condition_indices: indices of images that will be passed to the Generator in
                                                            order to generate fake images (for image-to-image
                                                            translation tasks). If set to None, the generator is fed
                                                            with random noise.
        :param (int) f1_k: `k` param of precision/recall metric (default is 3)
        """
        gen_dataset = gen_dataset.dataset if isinstance(gen_dataset, Subset) else gen_dataset
        if hasattr(gen_dataset, 'transforms'):
            self.gen_transforms = gen_dataset.transforms
        else:
            raise NotImplementedError('gen_dataset should expose image transforms (to invert them before entering '
                                      'ImageNET classifier to extract embeddings')
        # Save the dataset used by the generator in instance
        self.dataset = gen_dataset
        # Define metric calculators
        self.calculators = {
            'fid': FID(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                       batch_size=batch_size, device=device),
            'is': IS(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                     batch_size=batch_size, device=device),
            'f1': F1(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                     batch_size=batch_size, device=device),
            'ssim': SSIM(n_samples=n_samples, batch_size=batch_size, c_img=ssim_c_img, device=device),
        }
        # Save args
        self.target_index = target_index
        self.condition_indices = condition_indices
        self.z_dim = z_dim
        self.f1_k = f1_k

    def evaluate(self, gen: nn.Module, metric_name: Optional[str] = None, show_progress: bool = True) \
            -> Dict[str, float]:
        """
        Evaluate the generator's current state and return a `dict` with metric names as keys and evaluation results as
        values.
        :param (nn.Module) gen: the generator network as a `torch.nn.Module` object
        :param (optional) metric_name: the name of the evaluation metric to be applied
        :param (bool) show_progress: set to True to have the progress of evaluation metrics displayed (using `tqdm` lib)
        :return: if :attr:`metric` is `None` then a `dict` of all available metrics is returned, only the given metric
                 is returned otherwise
        """
        # Set generator in evaluation mode
        gen = gen.eval()
        metrics_dict = {}
        with torch.no_grad():
            for metric_name in self.calculators.keys() if not metric_name else (metric_name,):
                # Clear dataset transforms
                if metric_name != 'ssim':
                    self.dataset.transforms = None
                # Evaluate model
                metric = self.calculators[metric_name](
                    self.dataset, gen=gen, gen_transforms=self.gen_transforms, target_index=self.target_index,
                    condition_indices=self.condition_indices, z_dim=self.z_dim, skip_asserts=True,
                    show_progress=show_progress, k=self.f1_k
                )
                # Reset dataset transforms
                if metric_name != 'ssim':
                    self.dataset.transforms = self.gen_transforms
                # Unpack metrics
                if 'f1' == metric_name:
                    metrics_dict['f1'], metrics_dict['precision'], metrics_dict['recall'] = \
                        tuple(map(lambda _m: _m.item(), metric))
                else:
                    metrics_dict[metric_name] = metric.item()
        # Return metrics dict
        return metrics_dict
