import json
import os
import re
import sys
from shutil import copyfile
from time import sleep
from typing import Optional, Tuple

import click
from PIL import Image, UnidentifiedImageError
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, Compose

from datasets.deep_fashion import ICRBDataset
from datasets.samplers import ResumableRandomSampler
from utils.command_line_logger import CommandLineLogger
from utils.data import ManualSeedReproducible
from utils.dep_free import get_tqdm
from utils.filesystems.gdrive import GDriveDataset
from utils.filesystems.local import LocalCapsule, LocalFilesystem, LocalFolder
from utils.ifaces import ResumableDataLoader, FilesystemFolder
from utils.plot import squarify_img
from utils.string import to_human_readable
from utils.train import train_test_split


class PixelDTDataset(Dataset, GDriveDataset):
    """
    PixelDTDataset Class:
    This class is used to define the way LookBook dataset is accessed for the purpose of pixel-wise domain transfer
    (PixelDT).
    """

    # Dataset name is the name of the folder in Google Drive under which dataset's "Img.zip" file exists
    DatasetName = 'LookBook'

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 log_level: str = 'info'):
        """
        PixelDTDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `items_dt_info.json` is not present inside dataset's (local) root
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
        self.img_dir_path = f'{self.root}/Img'
        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')
        # Load item info
        self.items_dt_info_path = f'{self.img_dir_path}/items_dt_info.json'
        if not os.path.exists(self.items_dt_info_path):
            self.logger.error(f'items info file not found in image directory (tried: {self.items_dt_info_path})')
            sleep(0.5)
            if click.confirm('Do you want to scrape the dataset now?', default=True):
                PixelDTScraper.run()
        if not os.path.exists(self.items_dt_info_path):
            raise FileNotFoundError(f'{self.items_dt_info_path} not found in image directory')
        with open(self.items_dt_info_path) as fp:
            self.items_dt_info = json.load(fp)
        # Save benchmark info
        self.dt_image_pairs = self.items_dt_info['dt_image_pairs']
        self.dt_image_pairs_count = self.items_dt_info['dt_image_pairs_count']
        self.logger.debug(f'Found {to_human_readable(self.dt_image_pairs_count)} image pairs in dataset')
        # Save transforms
        self._transforms = None
        self.transforms = image_transforms

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        self._transforms = t if t else transforms.ToTensor()

    def index_to_paths(self, index: int) -> Tuple[str, str]:
        """
        Given an image-pair index it returns the file paths of the pair's images.
        :param (int) index: image-pair's index
        :return: a tuple containing (image_1_path, image_2_path), where lll file paths are absolute
        """
        image_1_path, image_2_path = self.dt_image_pairs[index]
        return f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_2_path}'

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param (int) index: integer with the current image index that we want to read from disk
        :return: a tuple containing the images from domain A (human) and B (product), each as a torch.Tensor object
        """
        paths_tuple = self.index_to_paths(index)
        # Fetch images
        try:
            img_s = Image.open(paths_tuple[0])
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {paths_tuple[0]})')
            return self.__getitem__(index + 1)
        img_t_path = paths_tuple[1]
        try:
            img_t = Image.open(img_t_path)
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {img_t_path})')
            return self.__getitem__(index + 1)
        # Apply transforms
        if self.transforms:
            img_s = self.transforms(img_s)
            img_t = self.transforms(img_t)
        return img_s, img_t

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.dt_image_pairs_count

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
        return ICRBDataset.get_image_transforms(target_shape=target_shape, target_channels=target_channels,
                                                norm_mean=norm_mean if norm_mean else PixelDTDataset.NormalizeMean,
                                                norm_std=norm_std if norm_std else PixelDTDataset.NormalizeStd)


class PixelDTDataloader(DataLoader, ResumableDataLoader, ManualSeedReproducible):
    """
    PixelDTDataloader Class:
    This class implement torch.utils.data.DataLoader and is used to access the underlying LookBook dataset with all the
    automation that PyTorch provides.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 target_shape: Optional[int] = None, target_channels: Optional[int] = None,
                 norm_mean: Optional[float] = None, norm_std: Optional[float] = None, batch_size: int = 8,
                 shuffle: bool = True, seed: int = 42, pin_memory: bool = True, splits: Optional[list] = None,
                 log_level: str = 'info'):
        """
        PixelDTDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download/use
                                                             dataset from local or remote (Google Drive) filesystem
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param target_shape: the H and W in the tensor coming out of image transforms
        :param target_channels: the number of channels in the tensor coming out of image transforms
        :param norm_mean: mean (same for all channels) or None to use dataset's default mean normalization parameter
        :param norm_std: standard deviation (same for all channels) or None to use dataset's default std normalization
                         parameter
        :param batch_size: the number of images batch
        :param shuffle: set to True to have sampler shuffle indices when reaches the end
        :param seed: manual seed parameter of torch.Generator (used in dataset sampler)
        :param pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is more thoroughly
                           explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param splits: if not None performs training/testing sets split based on given :attr:`split` percentages
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        # Reproducibility
        seed = ManualSeedReproducible.manual_seed(seed)
        # Image transforms
        if image_transforms is None and target_shape is None and target_channels is None:
            image_transforms = transforms.Compose([transforms.ToTensor()])
        assert image_transforms is None or (target_channels is None and target_shape is None), \
            'Do not define image_transforms when target_channels and target_shape is not None'
        assert (target_channels is not None and target_shape is not None) or (target_channels is None and target_shape
                                                                              is None), \
            'target_channels and target_shape can either both be None or both set'
        if target_shape is not None:
            image_transforms = PixelDTDataset.get_image_transforms(target_shape, target_channels,
                                                                   norm_mean=norm_mean, norm_std=norm_std)
        # Create dataset instance based on the transforms
        _dataset = PixelDTDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root,
                                  image_transforms=image_transforms, log_level=log_level)
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
        super(PixelDTDataloader, self).__init__(dataset=_training_set, batch_size=batch_size, sampler=self._sampler,
                                                pin_memory=pin_memory)

    def get_state(self) -> dict:
        return self._sampler.get_state()

    def set_state(self, state: dict) -> None:
        return self._sampler.set_state(state)


class PixelDTScraper:
    """
    PixelDTScraper Class:
    This class is used to scrape LookBook dataset's images for the purpose of pixel-wise domain transfer (PixelDT).
    """

    def __init__(self, root: str = '/data/Datasets/LookBook', log_level: str = 'info'):
        """
        LookBookScraper class constructor.
        :param (str) root: LookBook dataset's root directory path
        :param (str) log_level: see `utils.command_line_logger.CommandLineLogger`
        """
        self.tqdm = get_tqdm()
        self.logger = CommandLineLogger(log_level=log_level)
        self.initial_img_dir_path = f'{root}/ImgHQ'
        self.img_dir_path = f'{root}/Img'

        self.needs_initial_scraping = not os.path.exists(self.img_dir_path)
        if self.needs_initial_scraping and \
                click.confirm('Do you want to perform the initial scraping of the dataset now?', default=True):
            self.logger.info('[initial_scraping] STARTING')
            self.initial_scraping()
            self.logger.info('[initial_scraping] DONE')
            self.needs_initial_scraping = not os.path.exists(self.img_dir_path)

        self.items_count = 0 if self.needs_initial_scraping else len(next(os.walk(self.img_dir_path))[1])

        self.needs_initial_scraping_deep_fashion = self.items_count <= 8726
        if self.needs_initial_scraping_deep_fashion and \
                click.confirm('Do you want to perform the initial scraping of DeepFashion to expand LookBook with ' +
                              'more pairs now?', default=True):
            sleep(0.5)
            self.logger.info('[initial_scraping_deep_fashion] STARTING')
            sleep(0.5)
            self.initial_scraping_deep_fashion()
            self.logger.info('[initial_scraping_deep_fashion] DONE')
            self.needs_initial_scraping_deep_fashion = self.items_count <= 8726

    def initial_scraping(self, squarify: bool = True) -> None:
        """
        Transfers all images from initial image directory (as provided by the authors) into a directory structure where
        all the images of the same product id (pid) are placed into the same directory.
        :param squarify: set to True to have the original images expanded to square images and then resized to 256x256px
        """
        # Create output image directory
        if os.path.exists(self.img_dir_path):
            os.rmdir(self.img_dir_path)
        os.mkdir(self.img_dir_path)
        # Define regex
        img_regex = re.compile('PID(\d+)_CLEAN(\d)_IID(\d+).jpg')
        # Fetch image names as list
        initial_image_list = os.listdir(self.initial_img_dir_path)
        initial_image_list = sorted(initial_image_list)
        current_pid = None
        counter = None
        pid_dir_path = None
        for img in initial_image_list:
            # Get image data from filename
            pid, clean, iid = img_regex.findall('%s' % img)[0]
            pid = int(pid)
            # iid = int(iid)
            is_flat = '1' == clean
            # Initiate new Img/id_<8-digit_product_id> directory
            if pid != current_pid:
                # Before creating the new dir, check the previous contains flat.jpg and at least one other image
                if pid_dir_path is not None:
                    _, _, _files = next(os.walk(pid_dir_path))
                    assert len(_files) >= 2
                    assert 'flat.jpg' in _files
                # Create new directory
                pid_dir_path = f'{self.img_dir_path}/id_{str(pid).zfill(8)}'
                if not os.path.exists(pid_dir_path):
                    os.mkdir(pid_dir_path)
                self.logger.debug(f'initial_scraping(): New dir created at {pid_dir_path}')
                # Reset counter
                counter = 0
                current_pid = pid
            # Copy files
            src_img_filepath = f'{self.initial_img_dir_path}/{img}'
            dst_img_filename = str(counter) if not is_flat else 'flat'
            dst_img_filepath = f'{self.img_dir_path}/{os.path.basename(pid_dir_path)}/{dst_img_filename}.jpg'
            if squarify:
                result = squarify_img(src_img_filepath, target_shape=256, bg_color='white')
                result.save(dst_img_filepath, quality=95)
            else:
                copyfile(f'{src_img_filepath}', dst_img_filepath)
            # self.logger.debug(f'{src_img_filepath} --> {dst_img_filepath}')
            counter += 1

    def initial_scraping_deep_fashion(self) -> None:
        """
        Method to scrape DeepFashion ICRB dataset for items that contain image groups with flat.jpg images. These image
        groups can be copied over to LookBook Img root as new items since they can be used in a similar manner as the
        LookBook dataset's images. Transferred images are renamed to match naming style of LookBook item images.
        """
        # Define DeepFashion's Img root
        df_img_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark/Img'
        if not click.confirm(f'DeepFashion img root: {df_img_root}. Correct?', default=True) or \
                not os.path.exists(df_img_root):
            df_img_root = input('DeepFashion img root: ')
            assert os.path.exists(df_img_root), f'Provided df_img_root={df_img_root} NOT FOUND'
        items_info_path = f'{df_img_root}/items_info.json'
        assert os.path.exists(items_info_path), f'items_info.json NOT FOUND at DeepFashion Img root. Dataset should' + \
                                                f'have been scraped before. (tried path: {items_info_path})'

        # Load items info and look for image groups
        with open(items_info_path) as json_fp:
            items_info = json.load(json_fp)
        image_groups = items_info['image_groups']

        self.logger.debug(f'[initial_scraping_deep_fashion] Found {len(image_groups)} in total')

        # Get all useful image groups
        dt_groups = []
        dt_groups_count = 0
        for path, images in image_groups.items():
            flat_images = [_i for _i in images if _i.endswith('flat.jpg')]
            if 0 == len(flat_images):
                continue
            assert len(flat_images) == 1, f'len(flat_images)={len(flat_images)} > 1!'

            dt_groups.append({
                'src_path': f'{df_img_root}/{path}',
                'src_images': images,
            })
            dt_groups_count += 1

        self.logger.debug(f'[initial_scraping_deep_fashion] Found {dt_groups_count} out of {len(image_groups)}' +
                          f' items with a flat.jpg image.')
        with self.tqdm(total=dt_groups_count, file=sys.stdout) as progress_bar:
            # Copy images in LookBook under Img root
            pid = self.items_count
            for i, dt_group in enumerate(dt_groups):
                src_dir_path = dt_group['src_path']
                dst_dir_path = f'{self.img_dir_path}/id_{str((pid + i)).zfill(8)}'

                src_images = sorted(dt_group['src_images'], key=lambda _i: int(_i.split(sep='_', maxsplit=1)[0]))
                src_human_images = [_i for _i in src_images if _i.endswith('jpg') and not _i.endswith('flat.jpg')]
                src_flat_image = [_i for _i in src_images if _i.endswith('flat.jpg')][0]

                dst_human_images = [f'{dst_dir_path}/{_i}.jpg' for _i, _ in enumerate(src_human_images)]
                dst_flat_image = f'{dst_dir_path}/flat.jpg'

                src_human_images = [f'{src_dir_path}/{_i}' for _i in src_human_images]
                src_human_images = ['_'.join(_i.rsplit('/', 1)) for _i in src_human_images]
                src_flat_image = f'{src_dir_path}/{src_flat_image}'
                src_flat_image = '_'.join(src_flat_image.rsplit('/', 1))

                # Create destination folder and copy files
                if not os.path.exists(dst_dir_path):
                    os.mkdir(dst_dir_path)
                for _src, _dst in zip(src_human_images, dst_human_images):
                    copyfile(_src, _dst)
                copyfile(src_flat_image, dst_flat_image)

                # Stamp destination folder with DeepFashion data
                with open(f'{dst_dir_path}/.src', 'w') as fp:
                    fp.write(f'{src_flat_image.replace("/" + os.path.basename(src_flat_image), "")}\n')

                self.items_count += 1
                # self.logger.debug(f'[initial_scraping_deep_fashion] {dst_dir_path}: DONE')
                progress_bar.update()

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping LookBook images:
        Visits every item directory, process its images and saves image / pairs information a JSON file named
        `item_dt_info.json`.
        """
        id_dirs = next(os.walk(self.img_dir_path))[1]
        with self.tqdm(total=len(id_dirs), file=sys.stdout) as progress_bar:
            for id_dir in id_dirs:
                id_dir_path = f'{self.img_dir_path}/{id_dir}'
                images = os.listdir(id_dir_path)
                assert 'flat.jpg' in images
                images = sorted([_i for _i in images if _i.endswith('.jpg') and _i != 'flat.jpg'],
                                key=lambda _i: int(_i.replace('.jpg', '')))
                dt_image_pairs = [(_i, 'flat.jpg') for _i in images]
                dt_info = {
                    'id': int(id_dir.replace('id_', '')),
                    'path': f'/{id_dir}',
                    'flat_images': [
                        'flat.jpg'
                    ],
                    'flat_images_count': 1,
                    'human_images': images,
                    'human_images_count': len(images),
                    'dt_image_pairs': dt_image_pairs,
                    'dt_image_pairs_count': len(dt_image_pairs),
                }

                with open(f'{id_dir_path}/item_dt_info.json', 'w') as json_fp:
                    json.dump(dt_info, json_fp, indent=4)
                # self.logger.debug(f'{id_dir_path}: [DONE]')
                progress_bar.update()

    def backward(self) -> None:
        """
        Method for completing a backward pass in scraping LookBook images:
        Similar to DeepFashion scraper's backward pass, recursively visits all directories under image root merging
        information saved in JSON files found inside children directories.
        """
        # Initialize aggregator
        id_dirs = next(os.walk(self.img_dir_path))[1]
        items_dt_info = {
            'id': 'Img',
            'path': '',
            'flat_images': [],
            'flat_images_count': 0,
            'human_images': [],
            'human_images_count': 0,
            'dt_image_pairs': [],
            'dt_image_pairs_count': 0,
        }
        # Start merging
        with self.tqdm(total=len(id_dirs), file=sys.stdout) as progress_bar:
            for id_dir in id_dirs:
                id_dir_path = f'{self.img_dir_path}/{id_dir}'
                item_dt_info_path = f'{id_dir_path}/item_dt_info.json'
                assert os.path.exists(item_dt_info_path), f'item_dt_info_path={item_dt_info_path}: NOT FOUND'
                with open(item_dt_info_path) as json_fp:
                    item_dt_info = json.load(json_fp)

                # Prefix images
                file_prefix = item_dt_info['path'].lstrip('/')
                for _i, _name in enumerate(item_dt_info['flat_images']):
                    item_dt_info['flat_images'][_i] = f'{file_prefix}/{_name}'
                for _i, _name in enumerate(item_dt_info['human_images']):
                    item_dt_info['human_images'][_i] = f'{file_prefix}/{_name}'
                for _i, _pair in enumerate(item_dt_info['dt_image_pairs']):
                    item_dt_info['dt_image_pairs'][_i][0] = f'{file_prefix}/{_pair[0]}'
                    item_dt_info['dt_image_pairs'][_i][1] = f'{file_prefix}/{_pair[1]}'

                # Merge item in aggregated items info
                items_dt_info['flat_images'] += item_dt_info['flat_images']
                items_dt_info['flat_images_count'] += item_dt_info['flat_images_count']
                items_dt_info['human_images'] += item_dt_info['human_images']
                items_dt_info['human_images_count'] += item_dt_info['human_images_count']
                items_dt_info['dt_image_pairs'] += item_dt_info['dt_image_pairs']
                items_dt_info['dt_image_pairs_count'] += item_dt_info['dt_image_pairs_count']

                progress_bar.update()

        with open(f'{self.img_dir_path}/items_dt_info.json', 'w') as json_fp:
            json.dump(items_dt_info, json_fp, indent=4)

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True) -> None:
        """
        Entry point of class.
        :param forward_pass: set to True to run scraper's forward pass (create item_dt_info.json files in item dirs)
        :param backward_pass: set to True to run scraper's backward pass (recursively merge items JSON files)
                              Note: if :attr:`forward_pass` is set to True, then :attr:`backward_pass` will be True.
        """
        scraper = PixelDTScraper()
        scraper.logger.info(f'SCRAPE DIR = {scraper.img_dir_path}')
        if forward_pass:
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


if __name__ == '__main__':
    if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
        PixelDTScraper.run(forward_pass=True, backward_pass=True)

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _capsule = LocalCapsule(_local_gdrive_root)
    _fs = LocalFilesystem(ccapsule=_capsule)
    _groot = LocalFolder.root(capsule_or_fs=_fs).subfolder_by_name('Datasets')

    _pixel_dt = PixelDTDataset(dataset_fs_folder_or_root=_groot, log_level='debug')
    print(_pixel_dt.fetch_and_unzip())
