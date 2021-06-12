import os
from typing import Optional, Tuple

import click
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from h5py import Dataset as H5Dataset
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from datasets.deep_fashion import ICRBDataset
from utils.command_line_logger import CommandLineLogger
from utils.data import ResumableRandomSampler, ManualSeedReproducible
from utils.filesystems.gdrive import GDriveDataset
from utils.filesystems.local import LocalFolder, LocalCapsule
from utils.ifaces import ResumableDataLoader, FilesystemFolder
from utils.pytorch import ToTensorOrPass
from utils.string import to_human_readable
from utils.train import train_test_split


class Bags2ShoesDataset(Dataset, GDriveDataset):
    """
    Bags2ShoesDataset Class:
    This class is used to define the way Bags2Shoes dataset is accessed. The Bags2Shoes dataset consists of two distinct
    datasets, namely `handbags_64.hdf5` and `shoes_64.hdf5` containing 137K and 50K 64x64 images respectively.
    """

    # Dataset name is the name of the folder in Google Drive under which dataset's .hdf5 files live
    DatasetName = 'Bags2Shoes'

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 log_level: str = 'info'):
        """
        Bags2ShoesDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Instantiate `torch.utils.data.Dataset` class
        Dataset.__init__(self)
        # Instantiate `utils.filesystems.gdrive.GDriveDataset` class
        dataset_fs_folder = dataset_fs_folder_or_root if dataset_fs_folder_or_root.name == self.DatasetName else \
            dataset_fs_folder_or_root.subfolder_by_name(folder_name=self.DatasetName, recursive=True)
        GDriveDataset.__init__(self, dataset_fs_folder=dataset_fs_folder,
                               zip_filename=['handbags_64.hdf5', 'shoes_64.hdf5'])
        self.root = dataset_fs_folder.local_root
        # Initialize instance properties
        self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__)
        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')
        # Load hdf5 files
        #   - handbags_64.hdf5
        self.handbags_64_hdf5_path = os.path.join(self.root, 'handbags_64.hdf5')
        if not os.path.exists(self.handbags_64_hdf5_path):
            self.logger.error(
                f'handbags_64.hdf5 file not found in image directory (tried: {self.handbags_64_hdf5_path})')
            raise FileNotFoundError(f'{self.handbags_64_hdf5_path} not found in image directory')
        self.handbags_file = h5py.File(self.handbags_64_hdf5_path, 'r')
        self.handbags_dataset: H5Dataset
        self.handbags_dataset = self.handbags_file['imgs']
        self.handbags_total_images_count = self.handbags_dataset.len()
        self.logger.debug(f'Found {to_human_readable(self.handbags_total_images_count)} total images in the ' +
                          f'handbags_64 dataset')

        #   - shoes_64.hdf5
        self.shoes_64_hdf5_path = os.path.join(self.root, 'shoes_64.hdf5')
        if not os.path.exists(self.shoes_64_hdf5_path):
            self.logger.error(f'shoes_64.hdf5 file not found in image directory (tried: {self.shoes_64_hdf5_path})')
            raise FileNotFoundError(f'{self.shoes_64_hdf5_path} not found in image directory')
        self.shoes_file = h5py.File(self.shoes_64_hdf5_path, 'r')
        self.shoes_dataset: H5Dataset
        self.shoes_dataset = self.shoes_file['imgs']
        self.shoes_total_images_count = self.shoes_dataset.len()
        self.logger.debug(f'Found {to_human_readable(self.shoes_total_images_count)} total images in the ' +
                          f'shoes_64 dataset')
        # Save benchmark info
        self.total_images_count = min(self.handbags_total_images_count, self.shoes_total_images_count)
        self.logger.debug(f'Dataset\'s length (max): {to_human_readable(self.total_images_count)} images')
        # Save transforms
        self._transforms = None
        self.transforms = image_transforms

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        self._transforms = t if t is not None else transforms.ToTensor()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param (int) index: integer with the current image index that we want to read from disk
        :return: a tuple object containing one image from each domain (handbags and shoes respectively)
        """
        # Fetch handbag image
        handbag_image = Image.fromarray(self.handbags_dataset[index])
        # Fetch shoe image
        shoe_image = Image.fromarray(self.shoes_dataset[index])
        # Apply transforms & return
        handbag_image = self.transforms(handbag_image)
        shoe_image = self.transforms(shoe_image)
        return handbag_image, shoe_image

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in the In-shop Clothes Retrieval Benchmark.
        :return: the maximum length between the ones of each domain dataset
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
        norm_mean = Bags2ShoesDataset.NormalizeMean if norm_mean is None else norm_mean
        norm_std = Bags2ShoesDataset.NormalizeStd if norm_std is None else norm_std
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


class Bags2ShoesDataloader(DataLoader, ResumableDataLoader, ManualSeedReproducible):
    """
    Bags2ShoesDataloader Class.
    This class is used to load and access Bags2Shoes dataset using PyTorch's DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder,
                 image_transforms: Optional[transforms.Compose] = None, target_shape: Optional[int] = None,
                 target_channels: Optional[int] = None, norm_mean: Optional[float] = None,
                 norm_std: Optional[float] = None, batch_size: int = 8, shuffle: bool = True,
                 seed: int = 42, pin_memory: bool = True, splits: Optional[list] = None, log_level: str = 'info'):
        """
        Bags2ShoesDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :param batch_size: the number of images batch
        :param shuffle: set to True to have sampler shuffle indices when reaches the end
        :param (int) seed: manual seed parameter of torch.Generator (used in dataset sampler)
        :param (bool) pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is thoroughly
                           explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param (optional) splits: if not None performs training/testing sets split based on given :attr:`split`
                                  percentages
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Reproducibility
        seed = Bags2ShoesDataloader.manual_seed(seed)
        # Image transforms
        if image_transforms is None and target_shape is None and target_channels is None:
            image_transforms = transforms.Compose([transforms.ToTensor()])
        assert image_transforms is None or (target_channels is None and target_shape is None), \
            'Do not define image_transforms when target_channels and target_shape is not None'
        assert (target_channels is not None and target_shape is not None) or (target_channels is None and target_shape
                                                                              is None), \
            'target_channels and target_shape can either both be None or both set'
        if target_shape is not None:
            image_transforms = Bags2ShoesDataset.get_image_transforms(target_shape, target_channels,
                                                                      norm_mean=norm_mean, norm_std=norm_std)
        # Create dataset instance with the given transforms
        _entire_dataset = Bags2ShoesDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root,
                                            image_transforms=image_transforms, log_level=log_level)
        # Perform train/test split
        if splits:
            _training_set, _test_set = train_test_split(_entire_dataset, splits=splits, seed=seed)
            _entire_dataset.logger.debug(f'Split dataset: len(training_set)={to_human_readable(len(_training_set))}, ' +
                                         f'len(test_set)={to_human_readable(len(_test_set))}')
        else:
            _training_set = _test_set = _entire_dataset
        self.test_set = _test_set
        # Create sample instance
        self._sampler = ResumableRandomSampler(data_source=_training_set, shuffle=shuffle, seed=seed,
                                               logger=_entire_dataset.logger)
        # Finally, instantiate dataloader
        super(Bags2ShoesDataloader, self).__init__(dataset=_training_set, batch_size=batch_size, sampler=self._sampler,
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


if __name__ == '__main__':
    # Init Google Drive stuff
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _groot = LocalFolder.root(LocalCapsule(_local_gdrive_root))
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Init Dataset
    _dl = Bags2ShoesDataloader(_datasets_groot, target_shape=64, target_channels=3, batch_size=1,
                               pin_memory=False, log_level='debug')

    # Print first image from each dataset
    _bag, _shoe = next(iter(_dl))
    _plt_transforms = transforms.Compose([ToTensorOrPass(), transforms.ToPILImage()])
    plt.imshow(_plt_transforms(_bag.squeeze(dim=0)))
    plt.show()
    plt.imshow(_plt_transforms(_shoe.squeeze(dim=0)))
    plt.show()
