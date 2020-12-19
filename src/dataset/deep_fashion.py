import json
import os
import shutil
import sys
from time import sleep
from typing import Optional, Tuple, Union

import click
from PIL import Image, UnidentifiedImageError
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.command_line_logger import CommandLineLogger
from utils.list import get_pairs, list_diff, join_lists
from utils.string import group_by_prefix
from utils.string import to_human_readable


class ICRBDataset(Dataset):
    """
    ICRBCrossPoseDataset Class:
    This class is used to define the way DeepFashion's In-shop Clothes Retrieval Benchmark (ICRB) dataset is accessed.
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, hq: bool = False):
        """
        ICRBCrossPoseDataset class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param hq: set to True to process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        """
        super(ICRBDataset, self).__init__()
        # Test if running inside Colab
        self.inside_colab = 'google.colab' in sys.modules
        if root.startswith('/data') and self.inside_colab:
            root = f'/content{root}'
        self.logger = CommandLineLogger(log_level='info')
        self.img_dir_path = f'{root}/Img{"HQ" if hq else ""}'
        self.items_info_path = f'{self.img_dir_path}/items_info.json'
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
        self.total_images_count = self.items_info['images_count']
        self.logger.debug(f'Found {to_human_readable(self.total_images_count)} total images in benchmark')
        # Save transforms
        self.transforms = image_transforms

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


class ICRBDataloader(DataLoader):

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, batch_size: int = 8, hq: bool = False, *args):
        """
        ICRBCrossPoseDataloader class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param batch_size: the number of images batch
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param args: argument list for torch.utils.data.Dataloader constructor
        """
        _dataset = ICRBDataset(root=root, image_transforms=image_transforms, hq=hq)
        super(ICRBDataloader, self).__init__(dataset=_dataset, batch_size=batch_size, *args)


class ICRBCrossPoseDataset(Dataset):
    """
    ICRBCrossPoseDataset Class:
    This class is used to define the way DeepFashion's In-shop Clothes Retrieval Benchmark (ICRB) dataset is accessed to
    retrieve cross-pose/scale image pairs.
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, pose: bool = False, hq: bool = False):
        """
        ICRBCrossPoseDataset class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param hq: set to True to process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param pose: set to True to have __getitem__ return target pose image as well, otherwise __getitem__ will return
                     an image pair without target pose for the second image in pair
        """
        super(ICRBCrossPoseDataset, self).__init__()
        # Test if running inside Colab
        self.inside_colab = 'google.colab' in sys.modules
        if root.startswith('/data') and self.inside_colab:
            root = f'/content{root}'
        self.logger = CommandLineLogger(log_level='info')
        self.img_dir_path = f'{root}/Img{"HQ" if hq else ""}'
        self.items_info_path = f'{self.img_dir_path}/items_posable_info.json'
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
        self.logger.debug(f'Found {to_human_readable(self.total_pairs_count)} posable image pairs from a total of' +
                          f' {to_human_readable(self.total_images_count)} posable images in benchmark')
        # Save transforms
        self.transforms = image_transforms
        # Save pose switch
        self.pose = pose

    def index_to_paths(self, index: int) -> Union[Tuple[str, str], Tuple[str, str, str, str]]:
        """
        Given a image-pair index it returns the file paths of the pair's images.
        :param index: image-pair's index
        :return: a tuple containing (image_1_path, image_2_path) if pose is set to False, a tuple containing
                 (image_1_path, pose_1_path, image_2_path, pose_2_path) otherwise. All file paths are absolute.
        """
        _real_index = index % self.real_pairs_count
        _swap = _real_index == index
        # Find pair's image paths
        image_1_path, image_2_path = self.items_info['posable_image_pairs'][_real_index]
        if _swap:
            image_1_path, image_2_path = image_2_path, image_1_path

        return (f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_2_path}') if not self.pose else (
            f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_1_path.replace(".jpg", "_IUV.png")}',
            f'{self.img_dir_path}/{image_2_path}', f'{self.img_dir_path}/{image_2_path.replace(".jpg", "_IUV.png")}'
        )

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param index: integer with the current image index that we want to read from disk
        :return: a tuple containing the images from domain A and B, each as a torch.Tensor object
        """
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
        target_pose_2 = None if not self.pose else self.transforms(target_pose_2)
        return (image_1, image_2) if not self.pose else (image_1, image_2, target_pose_2)

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.total_pairs_count


class ICRBCrossPoseDataloader(DataLoader):

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, batch_size: int = 8, hq: bool = False, *args):
        """
        ICRBCrossPoseDataloader class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param batch_size: the number of images batch
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param args: argument list for torch.utils.data.Dataloader constructor
        """
        _dataset = ICRBCrossPoseDataset(root=root, image_transforms=image_transforms, pose=True, hq=hq)
        super(ICRBCrossPoseDataloader, self).__init__(dataset=_dataset, batch_size=batch_size, *args)


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
        directory there will be a huge JSON file "items_info.json" containing the image pairs for the whole dataset.
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

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark', hq: bool = False):
        """
        ICRBScraper class constructor.
        :param root: DeepFashion benchmark's root directory path
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        """
        # Test if running inside Colab
        if root.startswith('/data') and os.path.exists('/content'):
            root = f'/content/{root}'
        self.logger = CommandLineLogger(log_level='info')
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
        with tqdm(total=26 + (8082 if forward_pass else 0), colour='yellow') as progress_bar:
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
        }, [_i for _i in list(os.walk(self.img_dir_path)) if '/id_' in _i[0]]))
        id_dirs_list = sorted(id_dirs_list, key=lambda _i: _i['id'])

        # Search for duplicates
        duplicates_list = {}
        for _i in range(len(id_dirs_list)):
            _id = id_dirs_list[_i]['id']
            _id_str = str(_id)
            while _i < len(id_dirs_list) - 1 and _id == id_dirs_list[_i + 1]['id']:
                if _id_str not in duplicates_list.keys():
                    duplicates_list[_id_str] = [{
                        'path': id_dirs_list[_i]['path'],
                        'images': id_dirs_list[_i]['images'],
                    }, ]
                duplicates_list[_id_str].append({
                    'path': id_dirs_list[_i + 1]['path'],
                    'images': id_dirs_list[_i + 1]['images'],
                })
                _i += 1

        # print(json.dumps(duplicates_list, indent=4))

        duplicates_count = len(duplicates_list.keys())
        self.logger.info(f'Found {duplicates_count} duplicate items. Starting resolution...')

        # Resolve duplicates
        with tqdm(total=duplicates_count, colour='yellow') as progress_bar:
            for _id, _dirs in duplicates_list.items():
                _count_list = [len(_i['images']) for _i in _dirs]
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
        with tqdm(total=self.items_count + 100, colour='yellow') as progress_bar:
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
        with tqdm(total=self.items_count, colour='yellow') as progress_bar:
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
        for _i, _name in enumerate(_dict[f'{_key_prefix}images']):
            _dict[f'{_key_prefix}images'][_i] = f'{prefix}{_name}'
        for _k, _v in list(_dict[f'{_key_prefix}image_groups'].items()):
            _dict[f'{_key_prefix}image_groups'][f'{prefix}{_k}'] = _dict[f'{_key_prefix}image_groups'][_k].copy()
            del _dict[f'{_key_prefix}image_groups'][_k]
        if mode == 'posable_info':
            for _i, _pair in enumerate(_dict['posable_image_pairs']):
                _dict['posable_image_pairs'][_i][0] = f'{prefix}{_pair[0]}'
                _dict['posable_image_pairs'][_i][1] = f'{prefix}{_pair[1]}'

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
            with tqdm(total=root_dirs_count_at_depth[depth], colour='yellow') as progress_bar:
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


if __name__ == '__main__':
    if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
        ICRBScraper.run(forward_pass=True, backward_pass=True, hq=False)
