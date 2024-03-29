import collections
import json
import os
import shutil
import sys
import time
import zipfile
from time import sleep
from typing import Optional, Tuple, Union

import click
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import get_ipython
from PIL import Image, UnidentifiedImageError
from h5py import Dataset as H5Dataset
from matplotlib import ticker
from scipy.interpolate import make_interp_spline
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm

from datasets.samplers import ResumableRandomSampler
from utils.command_line_logger import CommandLineLogger
from utils.data import ManualSeedReproducible
from utils.dep_free import get_tqdm
from utils.filesystems.gdrive import GDriveDataset
from utils.filesystems.local import LocalFolder, LocalCapsule
from utils.ifaces import ResumableDataLoader, FilesystemFolder
from utils.list import get_pairs, list_diff, join_lists
from utils.pytorch import ToTensorOrPass
from utils.string import group_by_prefix
from utils.string import to_human_readable
from utils.train import train_test_split


class ICRBDataset(Dataset, GDriveDataset):
    """
    ICRBDataset Class:
    This class is used to define the way DeepFashion's In-shop Clothes Retrieval Benchmark (ICRB) dataset is accessed.
    """

    # Dataset name is the name of the folder in Google Drive under which dataset's Img.zip lives
    DatasetName = 'In-shop Clothes Retrieval Benchmark'

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 hq: bool = False, log_level: str = 'info'):
        """
        ICRBDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param (bool) hq: set to True to process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Instantiate `torch.utils.data.Dataset` class
        Dataset.__init__(self)
        # Instantiate `utils.filesystems.gdrive.GDriveDataset` class
        dataset_fs_folder = dataset_fs_folder_or_root if dataset_fs_folder_or_root.name == self.DatasetName else \
            dataset_fs_folder_or_root.subfolder_by_name(folder_name=self.DatasetName, recursive=False)
        GDriveDataset.__init__(self, dataset_fs_folder=dataset_fs_folder, zip_filename='Img.zip')
        self.root = dataset_fs_folder.local_root
        # Initialize instance properties
        self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__)
        self.img_dir_path = f'{self.root}/Img{"HQ" if hq else ""}'
        self.items_info_path = f'{self.img_dir_path}/items_info.json'
        # Load item info
        if not os.path.exists(self.items_info_path):
            self.logger.error(f'items info file not found in image directory (tried: {self.items_info_path})')
            sleep(0.5)
            if click.confirm('Do you want to scrape the dataset now?', default=True):
                ICRBScraper.run(hq=hq)
        if not os.path.exists(self.items_info_path):
            raise FileNotFoundError(f'{self.items_info_path} not found in image directory')
        with open(self.items_info_path) as fp:
            self.items_info = json.load(fp)
        # Save benchmark info
        self.total_images_count = self.items_info['images_count']
        self.logger.debug(f'Found {to_human_readable(self.total_images_count)} total images in benchmark')
        # Save transforms
        self._transforms = None
        self.transforms = image_transforms

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        self._transforms = t if t else transforms.ToTensor()

    def __getitem__(self, index: int) -> Tensor:
        """
        Implements abstract Dataset::__getitem__() method.
        :param index: integer with the current image index that we want to read from disk
        :return: an image from ICRB dataset as a torch.Tensor object
        """
        # Fetch image
        image = Image.open(f'{self.img_dir_path}/{self.items_info["images"][index]}')
        # Apply transforms
        image = self.transforms(image)
        # if image.shape[0] != 3:
        #     image = image.repeat(3, 1, 1)
        return image

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in the In-shop Clothes Retrieval Benchmark.
        :return: integer
        """
        return self.total_images_count

    @staticmethod
    def get_image_transforms(target_shape: int, target_channels: int, norm_mean: Optional[float] = None,
                             norm_std: Optional[float] = None) -> Compose:
        """
        Get the torchvision transforms to apply to the dataset based on default normalization parameters
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :return: a torchvision.transforms.Compose object with the transforms to apply to each dataset image
        """
        assert target_channels in [1, 3]
        norm_mean = ICRBDataset.NormalizeMean if norm_mean is None else norm_mean
        norm_std = ICRBDataset.NormalizeStd if norm_std is None else norm_std
        transforms_list = [
            transforms.Resize(target_shape),
            transforms.CenterCrop(target_shape),
            # transforms.RandomHorizontalFlip(),
        ]
        if target_channels == 1:
            transforms_list.append(transforms.Grayscale(1))
        transforms_list += [
            ToTensorOrPass(renormalize=False),
            transforms.Normalize(mean=tuple(np.ones(target_channels) * norm_mean),
                                 std=tuple(np.ones(target_channels) * norm_std))
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def get_root_prefix(default_prefix: Optional[str] = None) -> str:
        """
        Get default root prefix based on execution environment and given :attr:`default_prefix`
        :param (optional) default_prefix: if provided no automatic detection occurs
        :return: an str object containing the path prefix with the last "/" stripped
        """
        if not default_prefix:
            # Check if running inside Colab or Kaggle (auto prefixing)
            if 'google.colab' in sys.modules or 'google.colab' in str(get_ipython()) or 'COLAB_GPU' in os.environ:
                default_prefix = 'colab'
            elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                default_prefix = 'kaggle'
        # Find prefix path
        if default_prefix == 'colab':
            return '/content'
        elif default_prefix == 'kaggle':
            return '/kaggle/working'

        return default_prefix.rstrip('/') if default_prefix else ''


class ICRBDataloader(DataLoader, ResumableDataLoader, ManualSeedReproducible):
    """
    ICRBDataloader Class:
    This class is used to load and access DeepFashion's In-Shop Clothes Retrieval Benchmark's dataset using PyTorch's
    DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder,
                 image_transforms: Optional[transforms.Compose] = None, target_shape: Optional[int] = None,
                 target_channels: Optional[int] = None, norm_mean: Optional[float] = None,
                 norm_std: Optional[float] = None, batch_size: int = 8, hq: bool = False, shuffle: bool = True,
                 seed: int = 42, pin_memory: bool = True, splits: Optional[list] = None, log_level: str = 'info'):
        """
        ICRBDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :param batch_size: the number of images batch
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param shuffle: set to True to have sampler shuffle indices when reaches the end
        :param (int) seed: manual seed parameter of torch.Generator (used in dataset sampler)
        :param (bool) pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is thoroughly
                           explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param (optional) splits: if not None performs training/testing sets split based on given :attr:`split`
                                  percentages
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Reproducibility
        seed = ICRBDataloader.manual_seed(seed)
        # Image transforms
        if image_transforms is None and target_shape is None and target_channels is None:
            image_transforms = transforms.Compose([transforms.ToTensor()])
        assert image_transforms is None or (target_channels is None and target_shape is None), \
            'Do not define image_transforms when target_channels and target_shape is not None'
        assert (target_channels is not None and target_shape is not None) or (target_channels is None and target_shape
                                                                              is None), \
            'target_channels and target_shape can either both be None or both set'
        if target_shape is not None:
            image_transforms = ICRBDataset.get_image_transforms(target_shape, target_channels,
                                                                norm_mean=norm_mean, norm_std=norm_std)
        # Create dataset instance with the given transforms
        _dataset = ICRBDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, image_transforms=image_transforms,
                               hq=hq, log_level=log_level)
        # Perform train/test split
        if splits:
            _training_set, _test_set = train_test_split(_dataset, splits=splits, seed=seed)
            _dataset.logger.debug(f'Split dataset: len(training_set)={to_human_readable(len(_training_set))}, ' +
                                  f'len(test_set)={to_human_readable(len(_test_set))}')
        else:
            _training_set = _test_set = _dataset
        self.test_set = _test_set
        # Create sample instance
        self._sampler = ResumableRandomSampler(data_source=_training_set, shuffle=shuffle, seed=seed,
                                               logger=_dataset.logger)
        # Finally, instantiate dataloader
        super(ICRBDataloader, self).__init__(dataset=_training_set, batch_size=batch_size, sampler=self._sampler,
                                             pin_memory=pin_memory)

    def get_state(self) -> dict:
        return self._sampler.get_state()

    def set_state(self, state: dict) -> None:
        # FIX: Skipping last batch size (interrupted before forward pass completed)
        if 'perm_index' in state.keys():
            state['perm_index'] -= self.batch_size
            if state['perm_index'] < 0:
                state['perm_index'] = self.__len__() + state['perm_index'] + 1
        return self._sampler.set_state(state)


class ICRBCrossPoseDataset(Dataset, GDriveDataset):
    """
    ICRBCrossPoseDataset Class:
    This class is used to define the way DeepFashion's In-shop Clothes Retrieval Benchmark (ICRB) dataset is accessed so
    as to retrieve cross-pose/scale image pairs.
    """

    def __init__(self, dataset_fs_folder_or_root, image_transforms: Optional[transforms.Compose] = None,
                 skip_pose_norm: bool = True, pose: bool = True, hq: bool = False, log_level: str = 'info'):
        """
        ICRBCrossPoseDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param skip_pose_norm: set to True to remove Normalize() transform from pose images' transforms
        :param hq: set to True to process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param pose: set to True to have __getitem__ return target pose image as well, otherwise __getitem__ will return
                     an image pair without target pose for the second image in pair
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Instantiate `torch.utils.data.Dataset` class
        Dataset.__init__(self)
        # Instantiate `utils.filesystems.gdrive.GDriveDataset` class
        dataset_fs_folder = dataset_fs_folder_or_root if dataset_fs_folder_or_root.name == ICRBDataset.DatasetName \
            else dataset_fs_folder_or_root.subfolder_by_name(folder_name=ICRBDataset.DatasetName, recursive=False)
        GDriveDataset.__init__(self, dataset_fs_folder=dataset_fs_folder, zip_filename='Img.zip')
        self.root = dataset_fs_folder.local_root
        # Initialize instance properties
        self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__)
        self.img_dir_path = f'{self.root}/Img{"HQ" if hq else ""}'
        self.items_info_path = f'{self.img_dir_path}/items_posable_info.json'
        self.tqdm = get_tqdm()
        # Load item info
        if not os.path.exists(self.items_info_path):
            self.logger.error(f'items info file not found in image directory (tried: {self.items_info_path})')
            sleep(0.5)
            if click.confirm('Do you want to scrape the dataset now?', default=True):
                ICRBScraper.run(hq=hq)
        if not os.path.exists(self.items_info_path):
            raise FileNotFoundError(f'{self.items_info_path} not found in image directory')
        with open(self.items_info_path, 'r') as fp:
            self.items_info = json.load(fp)
        # Save benchmark info
        self.total_images_count = self.items_info['posable_images_count']
        self.real_pairs_count = self.items_info["posable_image_pairs_count"]
        self.total_pairs_count = self.real_pairs_count * 2  # 2nd pair by exchanging images at input/output
        self.logger.debug(f'Found {to_human_readable(self.total_pairs_count)} posable image pairs from a total of ' +
                          f'{to_human_readable(self.total_images_count)} posable images in benchmark')
        # Save transforms
        self._transforms = None
        self.skip_pose_norm = skip_pose_norm
        self.transforms = image_transforms
        # Save pose switch
        self.pose = pose

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        if t:
            self._transforms = t
            self.pose_transforms = t if not self.skip_pose_norm else \
                Compose([_t for _t in t.transforms if type(_t) != transforms.Normalize])
        else:
            self._transforms = transforms.ToTensor()
            self.pose_transforms = transforms.ToTensor()

    def index_to_paths(self, index: int) -> Union[Tuple[str, str], Tuple[str, str, str, str]]:
        """
        Given an image-pair index it returns the file paths of the pair's images.
        :param index: image-pair's index
        :return: a tuple containing (image_1_path, image_2_path) if pose is set to False, a tuple containing
                 (image_1_path, pose_1_path, image_2_path, pose_2_path) otherwise. All file paths are absolute.
        """
        _real_index = index % self.real_pairs_count
        _swap = (_real_index == index)
        # Find pair's image paths
        image_1_path, image_2_path = self.items_info['posable_image_pairs'][_real_index]
        if _swap:
            image_1_path, image_2_path = image_2_path, image_1_path

        return (f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_2_path}') if not self.pose else (
            f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_1_path.replace(".jpg", "_IUV.png")}',
            f'{self.img_dir_path}/{image_2_path}', f'{self.img_dir_path}/{image_2_path.replace(".jpg", "_IUV.png")}'
        )

    # IMAGE_1 = None
    # IMAGE_2 = None
    # IMAGE_2P = None

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param index: integer with the current image index that we want to read from disk
        :return: a tuple containing the images from domain A and B, each as a torch.Tensor object
        """

        # if self.IMAGE_1 is None:
        #     paths_tuple = self.index_to_paths(78400)
        #     image_1 = Image.open(paths_tuple[0])
        #     image_2_path = paths_tuple[1] if not self.pose else paths_tuple[2]
        #     image_2 = Image.open(image_2_path)
        #     target_pose_2 = None if not self.pose else Image.open(paths_tuple[3])
        #     # Apply transforms
        #     self.IMAGE_1 = self.transforms(image_1)
        #     self.IMAGE_2 = self.transforms(image_2)
        #     self.IMAGE_2P = None if not self.pose else self.pose_transforms(target_pose_2)
        # return (self.IMAGE_1, self.IMAGE_2) if not self.pose else (self.IMAGE_1, self.IMAGE_2, self.IMAGE_2P)

        # Get image paths
        paths_tuple = self.index_to_paths(index)
        # Fetch images
        try:
            image_1 = Image.open(paths_tuple[0])
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {paths_tuple[0]})')
            return self.__getitem__(index + 1)
        image_2_path = paths_tuple[1] if not self.pose else paths_tuple[2]
        try:
            image_2 = Image.open(image_2_path)
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {image_2_path})')
            return self.__getitem__(index + 1)
        try:
            target_pose_2 = None if not self.pose else Image.open(paths_tuple[3])
        except UnidentifiedImageError:
            self.logger.critical(f'Pose image opening failed (path: {paths_tuple[3]}')
            return self.__getitem__(index + 1)
        # Apply transforms
        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        target_pose_2 = None if not self.pose else self.pose_transforms(target_pose_2)
        return (image_1, image_2) if not self.pose else (image_1, image_2, target_pose_2)

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.total_pairs_count


class ICRBCrossPoseDataloader(DataLoader, ResumableDataLoader, ManualSeedReproducible):

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder,
                 image_transforms: Optional[transforms.Compose] = None, target_shape: Optional[int] = None,
                 target_channels: Optional[int] = None, norm_mean: Optional[float] = None,
                 norm_std: Optional[float] = None, skip_pose_norm: bool = True, batch_size: int = 8, hq: bool = False,
                 shuffle: bool = True, pin_memory: bool = True, seed: int = 42, splits: Optional[list] = None,
                 log_level: str = 'info'):
        """
        ICRBCrossPoseDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :param skip_pose_norm: set to True to remove Normalize() transform from pose images' transforms
        :param batch_size: the number of images batch
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param shuffle: set to True to have sampler shuffle indices when reaches the end
        :param pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is more thoroughly
                           explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param (int) seed: manual seed parameter of torch.Generator (used in dataset sampler)
        :param (optional) splits: if not None performs training/testing sets split based on given :attr:`split`
                                  percentages
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Reproducibility
        seed = ICRBCrossPoseDataloader.manual_seed(seed)
        # Image transforms
        if image_transforms is None and target_shape is None and target_channels is None:
            image_transforms = transforms.Compose([transforms.ToTensor()])
        assert image_transforms is None or (target_channels is None and target_shape is None), \
            'Do not define image_transforms when target_channels and target_shape is not None'
        assert (target_channels is not None and target_shape is not None) or (target_channels is None and target_shape
                                                                              is None), \
            'target_channels and target_shape can either both be None or both set'
        if target_shape is not None:
            image_transforms = ICRBDataset.get_image_transforms(target_shape, target_channels,
                                                                norm_mean=norm_mean, norm_std=norm_std)
        # Create dataset instance based on the transforms
        _dataset = ICRBCrossPoseDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, hq=hq, log_level=log_level,
                                        image_transforms=image_transforms, skip_pose_norm=skip_pose_norm, pose=True)
        # Perform train/test split
        if splits:
            _training_set, _test_set = train_test_split(_dataset, splits=splits, seed=seed)
            _dataset.logger.debug(f'Split dataset: len(training_set)={to_human_readable(len(_training_set))}, ' +
                                  f'len(test_set)={to_human_readable(len(_test_set))}')
        else:
            _training_set = _test_set = _dataset
        self.test_set = _test_set
        # Create sample instance
        self._sampler = ResumableRandomSampler(data_source=_training_set, shuffle=shuffle, seed=seed,
                                               logger=_dataset.logger)
        # Finally, instantiate dataloader
        super(ICRBCrossPoseDataloader, self).__init__(dataset=_training_set, batch_size=batch_size,
                                                      sampler=self._sampler, pin_memory=pin_memory)

    def get_state(self) -> dict:
        return self._sampler.get_state()

    def set_state(self, state: dict) -> None:
        # FIX: Skipping last batch size (interrupted before forward pass completed)
        if 'perm_index' in state.keys():
            state['perm_index'] -= self.batch_size
            if state['perm_index'] < 0:
                state['perm_index'] = self.__len__() + state['perm_index'] + 1
        return self._sampler.set_state(state)


class FISBDataset(Dataset, GDriveDataset):
    """
        FISBDataset Class:
        This class is used to define the way DeepFashion's Fashion Image Synthesis Benchmark (ICRB) dataset is accessed.
        """

    # Dataset name is the name of the folder in Google Drive under which dataset's Img.zip lives
    DatasetName = 'Fashion Synthesis Benchmark'

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    ################################################################################################################
    ################################################# DEV LOGGING ##################################################
    ################################################################################################################
    # IMAGE_R = None
    ################################################################################################################

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 load_in_memory: bool = False, verbose: bool = True, log_level: str = 'info',
                 logger: Optional[CommandLineLogger] = None, min_color: Optional[str] = None):
        """
        FISBDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param (bool) load_in_memory: set to True to load the whole h5 file in memory
        :param (bool) verbose: set to False to disable messages regarding Dataset initialization etc. (defaults to True)
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        :param (optional) logger: `utils.command_line_logger.CommandLineLogger` or None to instantiate a new one
        :param (optional) min_color: hex code of the minimum background color (to filter out images with dark bg)
        """
        # Instantiate `torch.utils.data.Dataset` class
        Dataset.__init__(self)
        # Instantiate `utils.filesystems.gdrive.GDriveDataset` class
        dataset_fs_folder = dataset_fs_folder_or_root if dataset_fs_folder_or_root.name == self.DatasetName else \
            dataset_fs_folder_or_root.subfolder_by_name(folder_name=self.DatasetName, recursive=True)
        GDriveDataset.__init__(self, dataset_fs_folder=dataset_fs_folder, zip_filename='Img.h5')
        self.root = dataset_fs_folder.local_root
        # Initialize instance properties
        self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__) if not logger else logger

        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')

        # Load h5 file
        self.h5_path = os.path.join(self.root, 'Img.h5')
        if not os.path.exists(self.h5_path):
            self.logger.error(f'Img.h5 file not found in image directory (tried: {self.h5_path})')
            raise FileNotFoundError(f'{self.h5_path} not found in image directory')
        self.img_file = h5py.File(self.h5_path, 'r')
        self.img_dataset: H5Dataset
        if load_in_memory:
            self.img_dataset = self.img_file['ih'][:]
        else:
            self.img_dataset = self.img_file['ih']
        self.img_mean = self.img_file['ih_mean']
        self.img_mean_max = np.max(self.img_mean)
        self.total_images_count = self.img_dataset.shape[0]
        if verbose:
            self.logger.debug(f'Found {to_human_readable(self.total_images_count)} total images in the ' +
                              f'Fashion Synthesis Benchmark dataset')

        # Load crops.json file
        self.crops_json_path = os.path.join(self.root, 'crops.json')
        with open(self.crops_json_path, 'r') as json_fp:
            self.crops = json.load(json_fp)
            assert len(self.crops) == self.total_images_count, \
                f'self.crops has {len(self.crops)} instead of {self.total_images_count}: ERROR'

        # Load backgrounds.json file
        self.backgrounds_json_path = os.path.join(self.root, 'backgrounds.json')
        with open(self.backgrounds_json_path, 'r') as json_fp:
            self.backgrounds = json.load(json_fp)
            background_counts = sum([len(indices) for indices in self.backgrounds.values()])
            assert background_counts == self.total_images_count, \
                f'self.backgrounds has {background_counts} instead of {self.total_images_count}: ERROR'

        # Filter image by background color
        if min_color is not None:
            colors = sorted(self.backgrounds.keys())
            min_color = min_color[1:].upper() if min_color.startswith('#') else min_color.upper()
            min_color_index = [c[0] for c in enumerate(colors) if c[1] >= min_color]
            if len(min_color_index) > 0:
                min_color_index = min_color_index[0]
                self.indices = []
                for ci in range(min_color_index, len(colors)):
                    self.logger.debug(f'Adding indices from color "{colors[ci]}"')
                    self.indices += self.backgrounds[colors[ci]]
            else:
                self.logger.critical(f'No indices for minimum color "{min_color}"')
                self.indices = list(range(self.total_images_count))
        else:
            self.indices = list(range(self.total_images_count))
        self.min_color = min_color
        self.total_images_count = len(self.indices)
        if verbose:
            self.logger.debug(f'[COLOR_FILTER] Found {to_human_readable(self.total_images_count)} total images ' +
                              f'in the Fashion Synthesis Benchmark dataset')

        ################################################################################################################
        #################################################  DEV LOGGING  ################################################
        ################################################################################################################
        # self.IMAGE_R = torch.randn(3, 128, 128)
        ################################################################################################################

        # Save transforms
        self._transforms = None
        self.transforms = image_transforms

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        self._transforms = t if t else transforms.ToTensor()

    def __getitem__(self, index: int) -> Tensor:
        """
        Implements abstract Dataset::__getitem__() method.
        :param index: integer with the current image index that we want to read from disk
        :return: an image from ICRB dataset as a torch.Tensor object
        """
        ################################################################################################################
        ################################################# DEV LOGGING ##################################################
        ################################################################################################################
        # return self.IMAGE_R
        ################################################################################################################
        # Create PIL Image object
        real_index = self.indices[index]
        img = self.img_mean_max * self.img_dataset[real_index] + self.img_mean
        img = img / np.max(img)
        image = Image.fromarray((255 * img).astype(np.uint8).swapaxes(2, 0), 'RGB')
        # Apply transforms
        image = self.transforms(image)
        # Crop (costs on average less than 2ms)
        crop = self.crops[real_index]
        cropped_image = FISBDataset.crop_to_bounds(image, h_from=crop['h_from'], h_until=crop['h_until'],
                                                   w_from=crop['w_from'], w_until=crop['w_until'],
                                                   upscale_size=image.shape[2])
        # return torch.cat((image, cropped_image), dim=2)
        return torch.clamp(cropped_image, min=0.0, max=1.0)

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in the In-shop Clothes Retrieval Benchmark.
        :return: integer
        """
        return self.total_images_count

    @staticmethod
    def get_image_transforms(target_shape: int, target_channels: int, norm_mean: Optional[float] = None,
                             norm_std: Optional[float] = None) -> Compose:
        """
        Get the torchvision transforms to apply to the dataset based on default normalization parameters
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :return: a torchvision.transforms.Compose object with the transforms to apply to each dataset image
        """
        norm_mean = FISBDataset.NormalizeMean if norm_mean is None else norm_mean
        norm_std = FISBDataset.NormalizeStd if norm_std is None else norm_std
        return ICRBDataset.get_image_transforms(target_shape=target_shape, target_channels=target_channels,
                                                norm_mean=norm_mean, norm_std=norm_std)

    @staticmethod
    def get_root_prefix(default_prefix: Optional[str] = None) -> str:
        """
        Get default root prefix based on execution environment and given :attr:`default_prefix`
        :param (optional) default_prefix: if provided no automatic detection occurs
        :return: an str object containing the path prefix with the last "/" stripped
        """
        return ICRBDataset.get_root_prefix(default_prefix=default_prefix)

    @staticmethod
    def crop_to_bounds(t: torch.Tensor or np.ndarray, h_from: int = 0, h_until: int = -1, w_from: int = 0,
                       w_until: int = -1, upscale_size: Optional[int] = None) -> torch.Tensor or np.ndarray:
        """
        Crops given tensor at given bounding box.
        :param (torch.Tensor) t: input image as a torch.Tensor object with CxHxW order
        :param (int) h_from: starting crop heightwise (defaults to 0)
        :param (int) h_until: starting crop heightwise (defaults to -1)
        :param (int) w_from: starting crop widthwise (defaults to 0)
        :param (int) w_until: starting crop widthwise (defaults to -1)
        :param (int) h_until: target height and width of image
        :param (int) upscale_size: set to an int to have the method return an upscaled tensor matching the given shape
        :return: a torch.Tensor object containing the cropped image with CxHxW order
        """
        t_cropped = t[:, h_from:h_until, w_from:w_until]
        # Upscale if requested
        if upscale_size:
            upsample = torch.nn.Upsample(size=upscale_size, mode='bicubic', align_corners=True)
            if type(t_cropped) == np.ndarray:
                return upsample(torch.from_numpy(t_cropped).unsqueeze(0)).squeeze(0).numpy().astype(t_cropped.dtype)
            t_cropped = upsample(t_cropped.unsqueeze(0)).squeeze(0)
        return t_cropped

    def to_zip(self):
        __transforms = self.transforms
        self.transforms = None
        # Create a temp dir
        out_dir = f'{self.root}/Imgs_{self.min_color.lower().replace("#", "")}'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(f'{out_dir}.zip'):
            # Create zip file
            with zipfile.ZipFile(f'{out_dir}.zip', 'w', zipfile.ZIP_STORED) as zip_fp:
                # Write all images
                for img_i in tqdm(range(self.__len__())):
                    img_path = os.path.join(out_dir, f'{str(img_i).zfill(5)}.png')
                    # save_image(self.__getitem__(img_i), img_path)
                    zip_fp.write(img_path, os.path.basename(img_path))
        self.logger.info(f'Zipfile saved at: "{out_dir}.zip"')


class FISBDataloader(DataLoader, ResumableDataLoader, ManualSeedReproducible):
    """
    FISBDataloader Class.
    This class is used to load and access DeepFashion's Fashion Image Synthesis Benchmark dataset using PyTorch's
    DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder or FISBDataset, load_in_memory: bool = False,
                 image_transforms: Optional[transforms.Compose] = None, target_shape: Optional[int] = None,
                 target_channels: Optional[int] = None, norm_mean: Optional[float] = None,
                 norm_std: Optional[float] = None, batch_size: int = 8, shuffle: bool = True, verbose: bool = True,
                 seed: int = 42, pin_memory: bool = True, splits: Optional[list] = None,
                 log_level: str = 'info', logger: Optional[CommandLineLogger] = None, min_color: Optional[str] = None):
        """
        FISBDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem.
                                                             If instance of FISBDataset is given, then the process of
                                                             creating it will be bypassed.
        :param (bool) load_in_memory: set to True to load the whole h5 file in memory
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :param batch_size: the number of images batch
        :param shuffle: set to True to have sampler shuffle indices when reaches the end
        :param (bool) verbose: set to False to disable messages regarding Dataset initialization etc. (defaults to True)
        :param (int) seed: manual seed parameter of torch.Generator (used in dataset sampler)
        :param (bool) pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is thoroughly
                           explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param (optional) splits: if not None performs training/testing sets split based on given :attr:`split`
                                  percentages
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        :param (optional) logger: `utils.command_line_logger.CommandLineLogger` or None to instantiate a new one
        :param (optional) min_color: hex code of the minimum background color (to filter out images with dark bg)
        """
        self.locals = locals()
        del self.locals['self']
        del self.locals['__class__']
        # Reproducibility
        seed = FISBDataloader.manual_seed(seed)
        # Image transforms
        if image_transforms is None and target_shape is None and target_channels is None:
            image_transforms = transforms.Compose([transforms.ToTensor()])
        assert image_transforms is None or (target_channels is None and target_shape is None), \
            'Do not define image_transforms when target_channels and target_shape is not None'
        assert (target_channels is not None and target_shape is not None) or (target_channels is None and target_shape
                                                                              is None), \
            'target_channels and target_shape can either both be None or both set'
        if target_shape is not None:
            image_transforms = FISBDataset.get_image_transforms(target_shape, target_channels,
                                                                norm_mean=norm_mean, norm_std=norm_std)
        # Create dataset instance with the given transforms
        if type(dataset_fs_folder_or_root) == FISBDataset:
            _entire_dataset = dataset_fs_folder_or_root
        else:
            _entire_dataset = FISBDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, verbose=verbose,
                                          image_transforms=image_transforms, log_level=log_level, logger=logger,
                                          load_in_memory=load_in_memory, min_color=min_color)
        self._entire_dataset = _entire_dataset
        # Perform train/test split
        if splits:
            _training_set, _test_set = train_test_split(_entire_dataset, splits=splits, seed=seed)
            if verbose:
                _entire_dataset.logger.debug(f'Split dataset: len(training_set)={to_human_readable(len(_training_set))}'
                                             f', len(test_set)={to_human_readable(len(_test_set))}')
        else:
            _training_set = _test_set = _entire_dataset
        self.test_set = _test_set
        # Create sample instance
        self._sampler = ResumableRandomSampler(data_source=_training_set, shuffle=shuffle, seed=seed,
                                               logger=_entire_dataset.logger)
        # Finally, instantiate dataloader
        super(FISBDataloader, self).__init__(dataset=_training_set, batch_size=batch_size, sampler=self._sampler,
                                             pin_memory=pin_memory)
        self.dataset: FISBDataset

    def get_state(self) -> dict:
        return self._sampler.get_state()

    def set_state(self, state: dict) -> None:
        if 'perm_index' in state.keys():
            # FIX: Skipping last batch size (interrupted before forward pass completed)
            # state['perm_index'] -= self.batch_size
            if state['perm_index'] < 0:
                state['perm_index'] = self.__len__() + state['perm_index'] + 1
        return self._sampler.set_state(state)

    def update_batch_size(self, batch_size: int) -> 'FISBDataloader':
        """
        Update DataLoader's batch_size property. This (unfortunately) means re-initializing the entire object.
        :return: a new FISBDataloader instance with all the parameters untouched except batch_size.
        """
        self.locals['dataset_fs_folder_or_root'] = self._entire_dataset
        self.locals['batch_size'] = batch_size
        self.locals['verbose'] = False
        self.locals['logger'] = None
        return FISBDataloader(**self.locals)


class ICRBScraper:
    """
    ICRBScraper Class:
    This class is used to scrape DeepFashion's In-shop Clothes Retrieval Benchmark images and extract image pairs.
    Forward Pass:
        Goes into every item folder (folders that are named item ID "id_<0-padded-8-digit-ID>") and creates a JSON file
        named "item_info.json" with information about images and cross-pose/scale pairs
    Backward Pass:
        Aggregates information on every category folder (e.g. "MEN/Shirts_Polos") of the items contained in the category
        and creates a JSON file "items_info.json". Recursive execution. After backward passes completion at the Img root
        directory there will be a huge JSON file "items_info.json" containing the image pairs for the entire dataset.
    When complete, in the Img root folder there will be two big JSON files containing the following aggregated info:
        1) items_info.json
        - id (int or str): "Img"
        - path (str): "/"
        - images (list): all of benchmark's image paths
        - images_count (int): number of benchmark's images
        - image_groups (list):  ['group1_prefix_rel': [group1 images], ...]
        - image_groups_count (int): total number of image groups in benchmark
        2) items_posable_info.json
        - id (int or str): "Img"
        - path (str): "/"
        - posable_images (list): all posable (i.e. with the associate pose image, *_IUV.png, present) image paths
        - posable_images_count (int): usable number of benchmark's images
        - posable_image_pairs (list):  [(pair1_image1, pair1_image2), (pair2_image1, pair2_image2) ...]
        - posable_image_pairs_count (int): total number of posable image pairs in benchmark
        - posable_image_groups (list):  ['group1_prefix_rel': [group1 images], ...]
        - posable_image_groups_count (int): total number of posable image groups in benchmark
        (all paths are relative from Img folder under passed $root$ directory)
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark', hq: bool = False,
                 log_level: str = 'info'):
        """
        ICRBScraper class constructor.
        :param root: DeepFashion benchmark's root directory path
        :param (bool) hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        self.tqdm = get_tqdm()
        # Test if running inside Colab
        if root.startswith('/data') and os.path.exists('/content'):
            root = f'/content/{root}'
        self.logger = CommandLineLogger(log_level=log_level)
        self.img_dir_path = f'{root}/Img{"HQ" if hq else ""}'

        self.item_dirs = []
        with open(f'{root}/Anno/list_item_inshop.txt') as fp:
            file_lines = fp.readlines()
            self.items_count = int(file_lines[0].rstrip('\n'))
            for i in range(1, len(file_lines)):
                self.item_dirs.append(file_lines[i].rstrip('\n'))
        self.logger.info(f'total_items = {self.items_count}')

    @staticmethod
    def is_posable(test_image: str, item_root_path: Optional[str] = None) -> bool:
        # Possible pose keys: 'front' (1), 'side' (2), 'back' (3), 'full' (4), 'additional' (1-4 or 5), 'flat' (6)
        # where the number in the parentheses denotes the pose annotations:
        #   1: frontal view
        #   2: side view
        #   3: back view
        #   4: zoom-out view
        #   5: zoom-in view      <-- may or may not contain human (test on image border)
        #   6: stand-alone view  <-- no human, just clothes
        pose_key = test_image.split('_')[-1].replace('.jpg', '')
        if 'flat' == pose_key:
            return False

        if pose_key in ['front', 'side', 'back', 'full', 'additional']:
            return os.path.exists(f'{item_root_path}/{test_image.replace(".jpg", "_IUV.png")}')

        # if pose_key == 'additional':
        #     if not group_images:
        #         # no group images provided, assume *_additional.jpg contains human (so as to have more pairs)
        #         return True
        #
        #     """
        #     Check *_additional.jpg:
        #                   (0,41)  ---  (0,215)
        #                     |            |
        #                     |            |
        #                 (255,41) --- (255,215)
        #     Check for a slice of 2px around this border (this is the border of the real image) and compare this slice
        #     with the corresponding slices from 2 images from group that we know for sure that there exists human model
        #     in (e.g from *_front.jpg & *_side.jpg, or from *_front.jpg & *_back.jpg)
        return False

    @staticmethod
    def get_item_type(item_path_abs: str) -> int:
        """
        Get "type" of item (see return).
        :param item_path_abs: absolute item directory path
        :return: -1 if not item dir, 0 if usable item, 1 if duplicate, 2 if useless item (e.g. only one picture)
        """
        if not os.path.basename(item_path_abs).startswith('id_'):
            return -1
        if os.path.exists(f'{item_path_abs}/.duplicate') and os.path.isfile(f'{item_path_abs}/.duplicate'):
            return 1
        if os.path.exists(f'{item_path_abs}/.skip') and os.path.isfile(f'{item_path_abs}/.skip'):
            with open(f'{item_path_abs}/.skip') as skip_fp:
                reason = skip_fp.readline().rstrip('\n').split(': ', maxsplit=1)[-1]
            if reason == 'DUPLICATE':
                return 1
            if reason == 'POSABLE_IMAGE_SHORTAGE':
                return 2
        return 0

    def clear_info(self, forward_pass: bool = True) -> None:
        """
        Clear any non-image files found in any directory inside Img/ root directory of dataset's benchmark.
        :param forward_pass: set to True to delete files also in item directories (i.e. directories whose names are
                             "id_<8-digit zero-padded item ID>")
        """
        with self.tqdm(total=26 + (8082 if forward_pass else 0), file=sys.stdout) as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                if not forward_pass and os.path.basename(_root).startswith('id_'):
                    continue

                _non_image_files = [_f for _f in _files if not _f.endswith('.jpg') and not _f.endswith('.png')]
                if len(_non_image_files) > 0:
                    for _f in _non_image_files:
                        os.remove(f'{_root}/{_f}')
                progress_bar.update()

    def resolve_duplicate_items(self):
        """
        Method to perform duplicate items resolution:
        In ICRB some items have been placed to two or more categories and in some cases even split between those. We
        search for duplicates in this method and finding the items with duplicates we merge them and "stamp" directories
        whose image files have been softly-transferred (no deletion) to the merged directory. Stamping is performed by
        adding a ".duplicate" file in every directory of the conflict dirs except for the one that is the merge
        destination.
        """
        # Create a list that contains all ID directories [{'id': X, 'path': ".../id_000000X"}, ...]
        id_dirs_list = list(map(lambda _i: {
            'id': int(os.path.basename(_i[0]).replace('id_', '')),
            'path': _i[0],
            'images': [_f for _f in _i[2] if _f.endswith('.jpg')],
        }, [i for i in list(os.walk(self.img_dir_path)) if '/id_' in i[0]]))
        id_dirs_list = sorted(id_dirs_list, key=lambda _i: _i['id'])

        # Search for duplicates
        duplicates_list = {}
        for i in range(len(id_dirs_list)):
            _id = id_dirs_list[i]['id']
            _id_str = str(_id)
            while i < len(id_dirs_list) - 1 and _id == id_dirs_list[i + 1]['id']:
                if _id_str not in duplicates_list.keys():
                    duplicates_list[_id_str] = [{
                        'path': id_dirs_list[i]['path'],
                        'images': id_dirs_list[i]['images'],
                    }, ]
                duplicates_list[_id_str].append({
                    'path': id_dirs_list[i + 1]['path'],
                    'images': id_dirs_list[i + 1]['images'],
                })
                i += 1

        # print(json.dumps(duplicates_list, indent=4))
        duplicates_count = len(duplicates_list.keys())
        self.logger.info(f'Found {duplicates_count} duplicate items. Starting resolution...')

        # Resolve duplicates
        with self.tqdm(total=duplicates_count, file=sys.stdout) as progress_bar:
            for _id, _dirs in duplicates_list.items():
                _count_list = [len(i['images']) for i in _dirs]
                _target_images_count = max(_count_list)
                _target_index = _count_list.index(_target_images_count)

                # Move files from all dirs to target directory
                _target_dir = _dirs[_target_index]['path']
                for _dir_index, _dir_info in enumerate(_dirs):
                    if _dir_index == _target_index:
                        continue

                    _src_dir = _dir_info['path']
                    for _file in _dir_info['images']:
                        if not os.path.exists(f"{_target_dir}/{_file}"):
                            self.logger.debug(f'Copying file: {_src_dir}/{_file} --> {_target_dir}/{_file}')
                            shutil.copy(f"{_src_dir}/{_file}", _target_dir)

                        _pose_file = _file.replace('.jpg', '_IUV.png')
                        if os.path.exists(f"{_src_dir}/{_pose_file}") \
                                and not os.path.exists(f"{_target_dir}/{_pose_file}"):
                            self.logger.debug(f'Copying file: {_src_dir}/{_pose_file} --> {_target_dir}/{_pose_file}')
                            shutil.copy(f"{_src_dir}/{_pose_file}", _target_dir)

                    # Stamp src dir as duplicate
                    with open(f'{_src_dir}/.duplicate', 'w') as dup_fp:
                        dup_fp.write('\n'.join([f'Moved To: {_target_dir}',
                                                f'Reason: Target Dir had {_target_images_count} images compared to ' +
                                                f'{len(_dir_info["images"])} of this Dir', '']))
                progress_bar.update()

    def resolve_non_posable_items(self) -> None:
        """
        Method to perform non-posable items resolution:
        "Non-Posable" items are the ones that yield zero cross-pose image pairs. Upon finding such items we "stamp"
        them by adding a ".skip" file in item's directory.
        """
        non_posable_items = []
        with self.tqdm(total=self.items_count + 100, file=sys.stdout) as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                if '.duplicate' in _files:
                    progress_bar.update()
                    continue

                _json_files = [_f for _f in _files if _f.endswith('.json')]
                assert len(_dirs) * (len(_files) - len(_json_files)) == 0, \
                    "a dir should contain only sub-dirs of image and/or JSON files (or possibly a .duplicate file)"

                if os.path.basename(_root).startswith('id_'):
                    item_images = [_f for _f in os.listdir(_root) if _f.endswith('.jpg')]
                    item_image_groups = group_by_prefix(item_images, '_')

                    # Process each image group individually
                    _to_delete_keys = []
                    for group_prefix, group_images in item_image_groups.items():
                        posable_images = [_image for _image in group_images
                                          if self.__class__.is_posable(f'{group_prefix}_{_image}', _root)]
                        if len(posable_images) < 2:
                            # Remove group if yields no posable image pairs
                            _to_delete_keys.append(group_prefix)
                            continue

                    # Delete useless image groups
                    for _k in _to_delete_keys:
                        del item_image_groups[_k]

                    # Stamp item if has no usable groups
                    if 0 == len(item_image_groups.keys()):
                        with open(f'{_root}/.skip', 'w') as skip_fp:
                            skip_fp.write('\n'.join(['Reason: POSABLE_IMAGE_SHORTAGE', '']))
                        self.logger.debug(f"Item {os.path.basename(_root)} stamped")
                        non_posable_items.append(os.path.basename(_root))

                    progress_bar.update()

        if len(non_posable_items) > 0:
            self.logger.info(f'Found {len(non_posable_items)} non posable items, i.e. items having 0 cross-pose pairs')
            self.logger.info(', '.join(non_posable_items))

    def _forward_item(self, item_path_abs: str, non_posable: bool = False):
        """
        Processes single item. Creates item_info.json file inside $item_path_abs$ containing information about image
        groups and cross-pose/scale image pairs found in item directory.
        :param item_path_abs: item directory (absolute path)
        :param non_posable: set to True to avoid creating item_posable_info.json inside item directory
        :return:
        """
        item_images = [_f for _f in os.listdir(item_path_abs) if _f.endswith('.jpg')]
        item_image_groups = group_by_prefix(item_images, '_')
        item_posable_image_groups = item_image_groups.copy()
        item_posable_image_pairs = []

        if not non_posable:
            # Process each image group individually
            _non_posable_group_keys = []
            for group_prefix, group_images in item_posable_image_groups.items():
                posable_images = [_image for _image in group_images
                                  if self.__class__.is_posable(f'{group_prefix}_{_image}', item_path_abs)]
                if len(posable_images) < 2:
                    # Remove group if yields no posable image pairs
                    _non_posable_group_keys.append(group_prefix)
                    continue
                # Extract group pairs
                item_posable_image_pairs += get_pairs(posable_images, prefix=f'{group_prefix}_')

            # Delete useless image groups
            for _k in _non_posable_group_keys:
                del item_posable_image_groups[_k]

            # Stamp item if has no usable groups
            if 0 == len(item_posable_image_pairs):
                with open(f'{item_path_abs}/.skip', 'w') as skip_fp:
                    skip_fp.write('\n'.join(['Reason: POSABLE_IMAGE_SHORTAGE', '']))
                self.logger.critical(f"Item {os.path.basename(item_path_abs)} should had been skipped...")
                return

            # Save item information in JSON files
            posable_images = join_lists(*[list(map(lambda _i: f'{group_prefix}_{_i}', group_images))
                                          for group_prefix, group_images in item_posable_image_groups.items()])

            with open(f'{item_path_abs}/item_posable_info.json', 'w') as json_fp:
                json.dump({
                    'id': int(os.path.basename(item_path_abs).replace('id_', '')),
                    'path': item_path_abs.replace(self.img_dir_path, ''),
                    'posable_images': posable_images,
                    'posable_images_count': len(posable_images),
                    'posable_image_groups': item_posable_image_groups,
                    'posable_image_groups_count': len(item_posable_image_groups.keys()),
                    'posable_image_pairs': item_posable_image_pairs,
                    'posable_image_pairs_count': len(item_posable_image_pairs),
                }, json_fp, indent=4)

        with open(f'{item_path_abs}/item_info.json', 'w') as json_fp:
            json.dump({
                'id': int(os.path.basename(item_path_abs).replace('id_', '')),
                'path': item_path_abs.replace(self.img_dir_path, ''),
                'images': item_images,
                'images_count': len(item_images),
                'image_groups': item_image_groups,
                'image_groups_count': len(item_image_groups.keys()),
            }, json_fp, indent=4)
        # self.logger.debug(f'{item_path_abs}/item_info.json: [DONE]')

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping DeepFashion ICRB images:
        Visits every item directory, process its images and saves image / pairs information a JSON file, item_info.json.
        """
        processed_ids = []
        processed_count = 0
        with self.tqdm(total=self.items_count, file=sys.stdout) as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                _json_files = [_f for _f in _files if _f.endswith('.json')]
                assert len(_dirs) * (len(_files) - len(_json_files)) == 0, \
                    "a dir should contain only sub-dirs of image and/or JSON files"

                if os.path.basename(_root).startswith('id_'):
                    item_type = self.__class__.get_item_type(_root)
                    if item_type in [-1, 1]:
                        # Duplicate item directory found. Skip processing.
                        continue

                    if item_type not in [0, 2]:
                        self.logger.critical('Really unexpected path occurred!')
                        assert False

                    _non_posable = item_type == 2
                    self._forward_item(_root, _non_posable)
                    # self.logger.info(f'\tProcessed: {os.path.basename(_root)}')
                    processed_ids.append(os.path.basename(_root))
                    processed_count += 1
                    progress_bar.update()

        if processed_count != self.items_count:
            self.logger.error(f'processed_count != total_count ({processed_count} != {self.items_count})')
            self.logger.error(str(list_diff(self.item_dirs, processed_ids)))

    @staticmethod
    def _get_item_info(json_filepath_abs: str, mode: str = 'info') -> dict:
        assert os.path.exists(json_filepath_abs), f"json_filepath_abs not exists (tried: {json_filepath_abs})"
        with open(json_filepath_abs, 'r') as json_fp:
            item_or_items_info_dict = json.load(json_fp)

        # Create prefix for info_dict content
        prefix = os.path.basename(item_or_items_info_dict['path']) + '/'

        """
        Sample item_info.json:
        {
            "id": 80,
            "path": "/MEN/Denim/id_00000080",
            "images": [
                "01_1_front.jpg",
                ...
            ],
            "images_count": 4,
            "image_groups": {
                "01": [
                    "1_front.jpg",
                    ...
                ],
                ...
            },
            "image_groups_count": 1
        }
        
        Sample item_posable_info.json:
        {
            "id": 80,
            "path": "/MEN/Denim/id_00000080",
            "posable_images": [
                "01_1_front.jpg",
                ...
            ],
            "posable_images_count": 4,
            "posable_image_groups": {
                "01": [
                    "1_front.jpg",
                    ...
                ]
            },
            "posable_image_groups_count": 1,
            "posable_image_pairs": [
                [
                    "01_1_front.jpg",
                    "01_2_side.jpg"
                ],
                ...
            ],
            "posable_image_pairs_count": 6
        }
        """
        _keys = ['images', 'images_count', 'image_groups', 'image_groups_count'] if mode == 'info' else \
            ['posable_images', 'posable_images_count', 'posable_image_groups', 'posable_image_groups_count',
             'posable_image_pairs', 'posable_image_pairs_count']

        _dict = dict((k, item_or_items_info_dict[k]) for k in _keys if k in item_or_items_info_dict)

        # Prefix images
        _key_prefix = mode.replace('info', '')
        for i, _name in enumerate(_dict[f'{_key_prefix}images']):
            _dict[f'{_key_prefix}images'][i] = f'{prefix}{_name}'
        for _k, _v in list(_dict[f'{_key_prefix}image_groups'].items()):
            _dict[f'{_key_prefix}image_groups'][f'{prefix}{_k}'] = _dict[f'{_key_prefix}image_groups'][_k].copy()
            del _dict[f'{_key_prefix}image_groups'][_k]
        if mode == 'posable_info':
            for i, _pair in enumerate(_dict['posable_image_pairs']):
                _dict['posable_image_pairs'][i][0] = f'{prefix}{_pair[0]}'
                _dict['posable_image_pairs'][i][1] = f'{prefix}{_pair[1]}'

        return _dict

    def _backward_dir(self, depth: int, dir_abs: str, progress_bar: tqdm) -> None:
        _dirs = next(os.walk(dir_abs))[1]
        items_info_dict = {}
        items_posable_info_dict = {}
        for _dir in _dirs:
            _dir_abs = f'{dir_abs}/{_dir}'
            if os.path.exists(f'{_dir_abs}/.duplicate'):
                continue

            _dir_info_dict = self.__class__._get_item_info(f'{_dir_abs}/item{"s" if depth < 2 else ""}_info.json')
            if not items_info_dict:
                # Initialize merged dict
                items_info_dict = _dir_info_dict.copy()
            else:
                # Merge _dir_info_dict with current items_info_dict
                items_info_dict['images'] += _dir_info_dict['images']
                items_info_dict['images_count'] += _dir_info_dict['images_count']
                items_info_dict['image_groups'] = {**items_info_dict['image_groups'], **_dir_info_dict['image_groups']}
                items_info_dict['image_groups_count'] += _dir_info_dict['image_groups_count']

            if not os.path.exists(f'{_dir_abs}/.skip'):
                _dir_posable_info_dict = self.__class__._get_item_info(f'{_dir_abs}/item{"s" if depth < 2 else ""}' +
                                                                       f'_posable_info.json', mode='posable_info')
                if not items_posable_info_dict:
                    # Initialize merged dict
                    items_posable_info_dict = _dir_posable_info_dict.copy()
                else:
                    # Merge _dir_info_dict with current items_info_dict
                    items_posable_info_dict['posable_images'] += _dir_posable_info_dict['posable_images']
                    items_posable_info_dict['posable_images_count'] += _dir_posable_info_dict['posable_images_count']
                    items_posable_info_dict['posable_image_groups'] = {
                        **items_posable_info_dict['posable_image_groups'],
                        **_dir_posable_info_dict['posable_image_groups']
                    }
                    items_posable_info_dict['posable_image_groups_count'] += \
                        _dir_posable_info_dict['posable_image_groups_count']
                    items_posable_info_dict['posable_image_pairs'] += _dir_posable_info_dict['posable_image_pairs']
                    items_posable_info_dict['posable_image_pairs_count'] += \
                        _dir_posable_info_dict['posable_image_pairs_count']

            progress_bar.update()

        # Save items_info.json JSON file
        items_info_dict['id'] = os.path.basename(dir_abs)
        items_info_dict['path'] = dir_abs.replace(self.img_dir_path, '')
        with open(f'{dir_abs}/items_info.json', 'w') as json_fp:
            json.dump(items_info_dict, json_fp, indent=4)

        # Save items_posable_info.json JSON file
        if len(items_posable_info_dict.keys()) > 0:
            items_posable_info_dict['id'] = os.path.basename(dir_abs)
            items_posable_info_dict['path'] = dir_abs.replace(self.img_dir_path, '')
            with open(f'{dir_abs}/items_posable_info.json', 'w') as json_fp:
                json.dump(items_posable_info_dict, json_fp, indent=4)

    def backward(self, depth: int = 2) -> None:
        """
        Method for completing a backward pass in scraping DeepFashion ICRB images:
        Recursively visits all directories at given $depth$ and merges information saved in JSON files from children
        directories.
        Depths:
            0: <item dir level> (root: /, dirs: ["MEN", "WOMEN"])
            1 (e.g.): root: /MEN, dirs: ["Denim", "Jackets_Vests", ...]
            2 (e.g.): root: /MEN/Denim, dirs: ["id_00000080", "id_00000089", ...]
            3: <item dir level>  (benchmark's images lay there)
        :param depth: integer denoting current backward scraping level (supported: 0, 1, 2)
        """
        root_dirs_count_at_depth = {
            0: 2,
            1: 23,
            2: 7982,
            3: 0,  # at item level there should be no dirs, just image files
        }
        while depth >= 0:
            with self.tqdm(total=root_dirs_count_at_depth[depth], file=sys.stdout) as progress_bar:
                for _root, _dirs, _files in os.walk(self.img_dir_path):
                    _relative_root = _root.replace(self.img_dir_path, '')
                    _depth = _relative_root.count('/')
                    if _depth != depth:
                        continue

                    self._backward_dir(_depth, _root, progress_bar)
            depth -= 1

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True, hq: bool = False) -> None:
        """
        Entry point of class.
        :param forward_pass: set to True to run scraper's forward pass (create item_info.json files in item dirs)
        :param backward_pass: set to True to run scraper's backward pass (recursively merge items JSON files)
                              Note: if $forward_pass$ is set to True, then $backward_pass$ is also set to True.
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        """
        scraper = ICRBScraper(hq=hq)
        scraper.logger.info(f'SCRAPE DIR = {scraper.img_dir_path}')
        # Clear current files
        scraper.logger.info('[clear_info] STARTING')
        scraper.clear_info(forward_pass=forward_pass)
        scraper.logger.info('[clear_info] DONE')
        if forward_pass:
            # Resolve duplicate items
            scraper.logger.info('[resolve_duplicate_items] STARTING')
            scraper.resolve_duplicate_items()
            scraper.logger.info('[resolve_duplicate_items] DONE')
            # Resolve non-posable items
            scraper.logger.info('[resolve_non_posable_items] STARTING')
            scraper.resolve_non_posable_items()
            scraper.logger.info('[resolve_non_posable_items] DONE')
            # Forward pass
            scraper.logger.info('[forward] STARTING')
            scraper.forward()
            scraper.logger.info('[forward] DONE')
            backward_pass = True
        # Backward pass
        if backward_pass:
            scraper.logger.info('[backward] STARTING')
            scraper.backward()
            scraper.logger.info('[backward] DONE')
        scraper.logger.info('DONE')


class FISBScraper:
    """
    FISBScraper Class:
    This class is used to scrape DeepFashion's Fashion Image Synthesis Benchmark images and correct images. In
    particular the dataset was poorly constructed and therefore manual cropping was required for the vast majority of
    benchmark's images. To avoid this, this class fetches every image from the hdf5 file, pre-process it, crops it and
    then stores it back to the file.
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/Fashion Synthesis Benchmark', log_level: str = 'info'):
        """
        FISBScraper class constructor.
        :param root: DeepFashion benchmark's root directory path
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        self.tqdm = get_tqdm()
        # Test if running inside Colab
        if root.startswith('/data') and os.path.exists('/content'):
            root = f'/content/{root}'
        self.root = root
        self.logger = CommandLineLogger(log_level=log_level)

        # Open h5 file
        self.h5_path = os.path.join(self.root, 'Img.h5')
        if not os.path.exists(self.h5_path):
            self.logger.error(f'Img.h5 file not found in image directory (tried: {self.h5_path})')
            raise FileNotFoundError(f'{self.h5_path} not found in image directory')
        self.img_file = h5py.File(self.h5_path, 'r')
        self.img_dataset: H5Dataset
        self.img_dataset = self.img_file['ih']
        self.img_mean = self.img_file['ih_mean']
        self.img_mean_max = np.max(self.img_mean)
        self.total_images_count = self.img_dataset.shape[0]
        self.logger.debug(f'Found {to_human_readable(self.total_images_count)} total images in the ' +
                          f'Fashion Synthesis Benchmark dataset')
        self.transforms = FISBDataset.get_image_transforms(target_shape=128, target_channels=3)

        self.crops = None
        self.crops_json_path = os.path.join(self.root, 'crops.json')
        if os.path.exists(self.crops_json_path):
            with open(self.crops_json_path, 'r') as json_fp:
                self.crops = json.load(json_fp)
                assert len(self.crops) == self.total_images_count, \
                    f'self.crops has {len(self.crops)} instead of {self.total_images_count}: ERROR'

        self.backgrounds = None
        self.backgrounds_json_path = os.path.join(self.root, 'backgrounds.json')
        if os.path.exists(self.crops_json_path):
            with open(self.backgrounds_json_path, 'r') as json_fp:
                self.backgrounds = json.load(json_fp)
                background_counts = sum([len(indices) for indices in self.backgrounds.values()])
                assert background_counts == self.total_images_count, \
                    f'self.backgrounds has {background_counts} instead of {self.total_images_count}: ERROR'

    # noinspection PyUnusedLocal
    @staticmethod
    def get_background_color(image: Tensor, w_start: int or tuple = (10, 105), h_start: int = 5, w_band: int = 15,
                             h_band: int = 10, index: Optional[int] = None, bin_band: int = 5):
        # print(image.shape)
        _w_start = w_start[0] if type(w_start) == tuple else w_start
        image_slice = image[:, :, (_w_start - 1):(_w_start + w_band + 1)]
        image_slice_avg = torch.mean(image_slice, dim=0)
        image_slice_avg = torch.stack([image_slice_avg, image_slice_avg, image_slice_avg], dim=0)
        image_slice_avg[0, 0, 0] = torch.min(image)
        image_slice_avg[-1, -1, -1] = torch.max(image)
        image_slice_avg = ToTensorOrPass()(image_slice_avg)[:, h_start:(h_start + h_band), 1:-1]
        image_slice_avg_pil = transforms.ToPILImage()(image_slice_avg)
        # Get colors
        image_colors = image_slice_avg_pil.getcolors()
        # try:
        #     assert len(image_colors) == 1, f'len(image_colors)={len(image_colors)}: {image_colors}'
        # except AssertionError as e:
        #     if len(w_start) > 1:
        #         return FISBScraper.get_background_color(image=image, w_start=w_start[1:len(w_start)], h_start=h_start,
        #                                                 w_band=w_band, h_band=h_band, index=index, bin_band=bin_band)
        #     # print(str(e))
        #     # image = transforms.Compose([ToTensorOrPass(), transforms.ToPILImage()])(image)
        #     # plt.imshow(image)
        #     # plt.title(f'Index #{index}')
        #     # plt.show()
        #     # plt.imshow(image_slice_avg_pil)
        #     # plt.title(f'{image_colors}')
        #     # plt.show()
        image_colors = sorted(image_colors, key=lambda entry: entry[0])
        rgb_value = bin_band * round(image_colors[-1][-1][-1] / bin_band)
        return ('%02x%02x%02x' % (rgb_value, rgb_value, rgb_value)).upper()

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping DeepFashion FISB images:
        Open every image, process & crop it and save it back to the h5 file.
        """
        time.sleep(1.0)
        self.crops = []
        self.backgrounds = {}
        for index in self.tqdm(range(self.total_images_count), file=sys.stdout):
            # Fetch image
            img_h5 = self.img_dataset[index]
            img = self.img_mean_max * img_h5 + self.img_mean
            img = img / np.max(img)
            img = Image.fromarray((255 * img).astype(np.uint8).swapaxes(2, 0), 'RGB')
            # Apply transforms
            img = self.transforms(img)
            # Get cropping bounds
            #   - height
            crop_until_top = FISBScraper.should_crop_top(img, crop_shape=list(range(1, 20)))
            keep_from_h = crop_until_top + 1 if crop_until_top is not False else 0
            crop_from_bottom = FISBScraper.should_crop_bottom(img, target_shape=list(range(90, 127)))
            keep_until_h = crop_from_bottom - 1 if crop_from_bottom is not False else (img.shape[1] - 1)
            keep_h = keep_until_h - keep_from_h + 1
            diff_h = img.shape[1] - keep_h
            #   - width
            keep_from_w = diff_h // 2
            keep_until_w = img.shape[2] - diff_h // 2 + (diff_h % 2)
            crop_dict = {
                'h_from': keep_from_h,
                'h_until': keep_until_h,
                'w_from': keep_from_w,
                'w_until': keep_until_w,
                'upscale_size': img.shape[2]
            }
            self.crops.append(crop_dict)
            # Crop image and get background
            image = FISBDataset.crop_to_bounds(img, h_from=crop_dict['h_from'], h_until=crop_dict['h_until'],
                                               w_from=crop_dict['w_from'], w_until=crop_dict['w_until'],
                                               upscale_size=img.shape[2])
            background_color = self.__class__.get_background_color(image, index=index, bin_band=10)
            if background_color not in self.backgrounds.keys():
                self.backgrounds[background_color] = []
            self.backgrounds[background_color].append(index)

        # Save crops dict
        with open(self.crops_json_path, 'w') as json_fp:
            json.dump(self.crops, json_fp, indent=4)

        # Save crops dict
        with open(self.backgrounds_json_path, 'w') as json_fp:
            json.dump(self.backgrounds, json_fp, indent=4)

    def backward(self) -> None:
        """
        Perform backward pass.
        """
        if self.crops is None:
            raise RuntimeError('self.crops is None: Exiting now...')
        h_from_probs = {}
        h_until_probs = {}
        w_from_probs = {}
        w_until_probs = {}
        keys = ['h_from', 'h_until', 'w_from', 'w_until']
        # keys = ['h_from', 'h_until']
        for index in self.tqdm(range(self.total_images_count), file=sys.stdout):
            # Get crop data
            crops = self.crops[index]
            # Fill dicts
            for k in keys:
                k_value = crops[k]
                # Append to corresponding dict
                probs_dict = locals()[f'{k}_probs']
                if k_value not in probs_dict:
                    probs_dict[k_value] = 1 / self.total_images_count
                else:
                    probs_dict[k_value] += 1 / self.total_images_count

        # Set matplotlib params
        matplotlib.rcParams["font.family"] = 'JetBrains Mono'
        # Configure matplotlib for pretty plots
        plt.rcParams["font.family"] = 'JetBrains Mono'

        # Output probs
        for k in keys:
            probs_dict = locals()[f'{k}_probs']
            probs_dict = collections.OrderedDict(sorted(probs_dict.items()))
            # print(json.dumps(probs_dict, indent=4))
            x = [int(k) for k in probs_dict.keys()]
            x_smooth = np.linspace(min(x), max(x), 300)
            y = [float(v) for v in probs_dict.values()]
            # spl = make_interp_spline(x, 7, k=3)
            y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

            # Create a new figure
            plt.figure(figsize=(10, 5), dpi=300, clear=True)
            # Set data
            plt.plot(x_smooth, y_smooth, '-.', color='#2a9ceb')
            plt.plot(x, y, 'o', color='#1f77b4')
            # Set title
            plt_title = f'{k} Frequencies'
            plt_subtitle = f'FISBScraper::backward()'
            plt.suptitle(f'{plt_title}', y=0.97, fontsize=12, fontweight='bold')
            plt.title(f'{plt_subtitle}', pad=10., fontsize=10, )
            # Get PIL image
            plt.savefig(f'fisb_{k}_freqs.svg')
            # pil_img = pltfig_to_pil(plt.gcf())
            # pil_img.save(f'{k}_freqs.png')

        if self.backgrounds is None:
            raise RuntimeError('self.backgrounds is None: Exiting now...')
        colors = sorted(self.backgrounds.keys())
        indices_per_color = [len(self.backgrounds[color]) for color in colors]
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300, clear=True)
        # Set data
        x = range(len(colors))
        y = indices_per_color
        x_smooth = np.linspace(min(x), max(x), 300)
        y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
        ax.plot(x_smooth, y_smooth, '-.', color='#2a9ceb')
        ax.plot(x, y, 'o', color='#1f77b4')
        plt.xticks(x, labels=[f'#{c}' for c in colors], rotation=45)
        loc = ticker.MultipleLocator(base=2.0)
        ax.xaxis.set_major_locator(loc)
        # Set title
        plt_title = f'Image Background Frequencies'
        plt_subtitle = f'FISBScraper::backward()'
        plt.suptitle(f'{plt_title}', y=0.97, fontsize=12, fontweight='bold')
        plt.title(f'{plt_subtitle}', pad=10., fontsize=10, )
        # Get PIL image
        plt.savefig(f'fisb_background_freqs.svg')
        # pil_img = pltfig_to_pil(plt.gcf())
        # pil_img.save(f'background_freqs.png')
        # plt.show()

    @staticmethod
    def should_crop_top(t: torch.Tensor, crop_shape: Union[int, list] = 10) -> int or False:
        """
        Check whether the image in given tensor should be cropped, be checking if the area to be cropped is ones only.
        :param (torch.Tensor) t: input image as a torch.Tensor object with CxHxW order
        :param (int or list) crop_shape: crop height and width of image (or sequence of crop shapes for multiple checks)
        :return: the found target shape if one found, otherwise False is returned
        """
        if type(crop_shape) == int:
            crop_shape = (crop_shape,)
        crop_shape.sort(reverse=True)
        for cs in crop_shape:
            initial_shape = t.shape[1]
            final_shape = initial_shape - cs
            diff_shape = initial_shape - final_shape
            width_limits = (diff_shape // 2 + (diff_shape % 2), (diff_shape // 2 + (diff_shape % 2)))
            # Check if square is neutral (e.g. white or gray)
            cropped_area: torch.Tensor
            cropped_area = t[:, 0:cs, width_limits[0]:(initial_shape - width_limits[1])].clone()
            cropped_area[0, 0, 0] = -1.0
            cropped_area[-1, -1, -1] = 1.0
            cropped_area = ToTensorOrPass()(cropped_area)
            cropped_area = cropped_area[:, :, width_limits[0]:max(1, (cropped_area.shape[2] - width_limits[1]))]
            if sum([abs(cropped_area[c_i].min() - cropped_area[c_i].max()) for c_i in range(3)]) < 0.1 \
                    or (torch.allclose(cropped_area[0], cropped_area[1], rtol=1e-2) and
                        torch.allclose(cropped_area[1], cropped_area[2], rtol=1e-2)):
                return cs
        return False

    @staticmethod
    def should_crop_bottom(t: torch.Tensor, target_shape: Union[int, list] = 110) -> int or False:
        """
        Check whether the image in given tensor should be cropped, be checking if the area to be cropped is ones only.
        :param (torch.Tensor) t: input image as a torch.Tensor object with CxHxW order
        :param (int or list) target_shape: target height and width of image (or sequence of target shapes for multiple
                                            checks)
        :return: the found target shape if one found, otherwise False is returned
        """
        if type(target_shape) == int:
            target_shape = (target_shape,)
        for ts in target_shape:
            initial_shape = t.shape[1]
            diff_shape = initial_shape - ts
            width_limits = (diff_shape // 2, (diff_shape // 2 + (diff_shape % 2)))
            # Check if square is neutral (e.g. white or gray)
            cropped_area: torch.Tensor
            cropped_area = t[:, ts + 1:, width_limits[0]:(initial_shape - width_limits[1])].clone()
            cropped_area[0, 0, 0] = -1.0
            cropped_area[-1, -1, -1] = 1.0
            cropped_area = ToTensorOrPass()(cropped_area)
            cropped_area = cropped_area[:, :, width_limits[0]:(cropped_area.shape[2] - width_limits[1])]
            if sum([abs(cropped_area[c_i].min() - cropped_area[c_i].max()) for c_i in range(3)]) < 0.1 \
                    or (torch.allclose(cropped_area[0], cropped_area[1], rtol=1e-2) and
                        torch.allclose(cropped_area[1], cropped_area[2], rtol=1e-2)):
                return ts
        return False

    @staticmethod
    def crop_top(t: torch.Tensor or np.ndarray, crop_shape: int = 110, bounds_only: bool = False,
                 upscale_after: bool = False) -> torch.Tensor or np.ndarray:
        """
        Crops given tensor at target_shape (square crop).
        :param (torch.Tensor) t: input image as a torch.Tensor object with CxHxW order
        :param (int) crop_shape: height of crop area
        :param (bool) bounds_only: set to True to return cropping bounds only and not perform any actual operation on
                                   the inputs
        :param (bool) upscale_after: set to True to have the method return an upscaled tensor matching the dimensions
                                     of the input one
        :return: a torch.Tensor object containing the cropped image with CxHxW order
        """
        assert t.shape[1] == t.shape[2] and t.shape[1] > t.shape[0], 'check that ordering is CxHxW'
        # Perform the crop by centering in width ONLY
        initial_shape = t.shape[1]
        final_shape = initial_shape - crop_shape
        diff_shape = initial_shape - final_shape
        width_limits = (diff_shape // 2 + (diff_shape % 2), (diff_shape // 2 + (diff_shape % 2)))
        bounds = [crop_shape + 1, -1, width_limits[0], (initial_shape - width_limits[1])]
        if bounds_only:
            return bounds
        return FISBDataset.crop_to_bounds(t, bounds[0], bounds[1], bounds[2], bounds[3],
                                          upscale_size=initial_shape if upscale_after else None)

    @staticmethod
    def crop_bottom(t: torch.Tensor or np.ndarray, target_shape: int = 110, bounds_only: bool = False,
                    upscale_after: bool = False) -> torch.Tensor or np.ndarray:
        """
        Crops given tensor at target_shape (square crop).
        :param (torch.Tensor) t: input image as a torch.Tensor object with CxHxW order
        :param (int) target_shape: target height and width of image
        :param (bool) bounds_only: set to True to return cropping bounds only and not perform any actual operation on
                                   the inputs
        :param (bool) upscale_after: set to True to have the method return an upscaled tensor matching the dimensions
                                     of the input one
        :return: a torch.Tensor object containing the cropped image with CxHxW order
        """
        assert t.shape[1] == t.shape[2] and t.shape[1] > t.shape[0], 'check that ordering is CxHxW'
        # Perform the crop by centering in width ONLY
        initial_shape = t.shape[1]
        diff_shape = initial_shape - target_shape
        bounds = [0, target_shape, diff_shape // 2, (initial_shape - diff_shape // 2 - (diff_shape % 2))]
        if bounds_only:
            return bounds
        return FISBDataset.crop_to_bounds(t, bounds[0], bounds[1], bounds[2], bounds[3],
                                          upscale_size=initial_shape if upscale_after else None)

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True) -> None:
        """
        Entry point of class.
        :param forward_pass: set to True to run scraper's forward pass (create crops.json file in dataset's root)
        :param backward_pass: set to True to run scraper's backward pass (read crops.json and output stats)
                              Note: if $forward_pass$ is set to True, then $backward_pass$ is also set to True.
        """
        scraper = FISBScraper()
        scraper.logger.info(f'SCRAPE DIR = {scraper.root}')
        if forward_pass:
            # Forward pass
            scraper.logger.info('[forward] STARTING')
            time.sleep(0.5)
            scraper.forward()
            time.sleep(0.5)
            scraper.logger.info('[forward] DONE')
        # Backward pass
        if backward_pass:
            scraper.logger.info('[backward] STARTING')
            time.sleep(0.5)
            scraper.backward()
            time.sleep(0.5)
            scraper.logger.info('[backward] DONE')
        scraper.logger.info('DONE')


if __name__ == '__main__':
    # if click.confirm('Do you want to (re)scrape the ICRB dataset now?', default=True):
    #     ICRBScraper.run(forward_pass=True, backward_pass=True, hq=False)
    # if click.confirm('Do you want to (re)scrape the FISB dataset now?', default=True):
    #     FISBScraper.run(forward_pass=False, backward_pass=True)
    # exit(0)

    # Init Google Drive stuff
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _groot = LocalFolder.root(LocalCapsule(_local_gdrive_root))
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Init Dataset
    # _dl = FISBDataloader(_datasets_groot, target_shape=128, target_channels=3, batch_size=1,
    #                      pin_memory=False, log_level='debug', load_in_memory=False,
    #                      min_color='#f0f0f0')
    _ds = FISBDataset(_datasets_groot, log_level='debug', load_in_memory=False, min_color='#f0f0f0')
    _ds.to_zip()

    # # Index 54886 is dark
    # print(f'len(_dl.dataset)={len(_dl.dataset)}')
    # for _i in np.random.randint(0, len(_dl.dataset), 20):
    #     _img = _dl.dataset[_i]
    #     _plt_transforms = transforms.Compose([ToTensorOrPass(), transforms.ToPILImage()])
    #     _img = _plt_transforms(_img.squeeze(dim=0))
    #     _img.save(f'{str(_i).zfill(4)}.jpg')
    #     plt.imshow(_img)
    #     plt.title(f'Index #{str(_i).zfill(4)}')
    #     plt.show()

    ### Print first image from each dataset
    # from matplotlib import pyplot as plt
    # from utils.plot import create_img_grid, plot_grid
    # for i in np.random.randint(0, len(_dl.dataset), 10):
    #     # i = 76820
    #     print(i)
    #     _img = _dl.dataset[i]
    #     _imgs = [_img.clone(), ]
    #
    #     # Get cropping bounds
    #     #   - height
    #     _crop_until_top = FISBScraper.should_crop_top(_img, crop_shape=list(range(1, 20)))
    #     _keep_from_h = _crop_until_top + 1 if _crop_until_top is not False else 0
    #     _crop_from_bottom = FISBScraper.should_crop_bottom(_img, target_shape=list(range(107, 127)))
    #     _keep_until_h = _crop_from_bottom - 1 if _crop_from_bottom is not False else (_img.shape[1] - 1)
    #     _keep_h = _keep_until_h - _keep_from_h + 1
    #     _diff_h = _img.shape[1] - _keep_h
    #     #   - width
    #     _keep_from_w = _diff_h // 2
    #     _keep_until_w = _img.shape[2] - _diff_h // 2 + (_diff_h % 2)
    #
    #     # Crop
    #     try:
    #         _img = FISBDataset.crop_to_bounds(_img, h_from=_keep_from_h, h_until=_keep_until_h, w_from=_keep_from_w,
    #                                           w_until=_keep_until_w, upscale_size=_img.shape[2])
    #         _imgs.append(_img.clone())
    #     except RuntimeError as e:
    #         dataset_ptr: FISBDataset = _dl.dataset
    #         dataset_ptr.logger.error(str(e) + ' | ' + str((_keep_from_h, _keep_until_h, _keep_from_w, _keep_until_w)))
    #
    #     _plt_transforms = transforms.Compose([ToTensorOrPass(), transforms.ToPILImage()])
    #     if len(_imgs) == 1:
    #         pass
    #     else:
    #         _grid = create_img_grid(images=torch.stack(_imgs), ncols=len(_imgs), nrows=1)
    #         _img = plot_grid(grid=_grid, figsize=(len(_imgs), 1), footnote_l=f'index={i}', footnote_r='[CROPPED]')
    #         _img = ToTensorOrPass()(_img)
    #
    #     # Plot before/after
    #     plt.imshow(_plt_transforms(_img.squeeze(dim=0)))
    #     plt.show()

    # if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
    #     FISBScraper.run(forward_pass=False, backward_pass=True)
