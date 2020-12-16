import json
import os
import shutil
from typing import Optional, Tuple, Union

import click
from PIL import Image
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.command_line_logger import CommandLineLogger
from utils.list import get_pairs, list_diff, join_lists
from utils.string import group_by_prefix
from utils.string import to_human_readable


class InShopClothesRetrievalBenchmarkDataset(Dataset):
    """
    InShopClothesRetrievalBenchmarkDataset Class:
    This class is used to define the way DeepFashion's In-shop Clothes Retrieval Benchmark (ICRB) dataset is accessed.
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, pose: bool = False, hq: bool = False):
        """
        InShopClothesRetrievalBenchmarkDataset class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param hq: set to True to process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param pose: set to True to have __getitem__ return target pose image as well
        """
        super(InShopClothesRetrievalBenchmarkDataset, self).__init__()
        self.logger = CommandLineLogger(log_level='info')
        self.img_dir_path = f'{root}/Img{"HQ" if hq else ""}'
        self.items_info_path = f'{self.img_dir_path}/items_info.json'
        # Load item info
        if not os.path.exists(self.items_info_path):
            self.logger.error('items_info.json not found in image directory')
            if click.confirm('Do you want to scrape the dataset now?', default=True):
                InShopClothesRetrievalBenchmarkScraper.run(hq=hq)
        if not os.path.exists(self.items_info_path):
            raise FileNotFoundError('items_info.json not found in image directory')
        with open(self.items_info_path, 'r') as fp:
            self.items_info = json.load(fp)
        # Save benchmark info
        self.real_pairs_count = self.items_info["image_pairs_count"]
        self.total_pairs_count = self.real_pairs_count * 2  # 2nd pair by exchanging images at input/output of the model
        self.total_images_count = self.items_info["images_count"]
        self.logger.debug(f'Found {to_human_readable(self.total_pairs_count)} image pairs from a total of' +
                          f' {to_human_readable(self.total_images_count)} images in benchmark')
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
        image_1_path, image_2_path = self.items_info['image_pairs'][_real_index]
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
        image_1 = Image.open(paths_tuple[0])
        image_2 = Image.open(paths_tuple[1] if not self.pose else paths_tuple[2])
        target_pose_2 = None if not self.pose else Image.open(paths_tuple[3])
        # Apply transforms
        image_1 = self.transforms(image_1)
        image_2 = self.transforms(image_2)
        target_pose_2 = None if not self.pose else self.transforms(target_pose_2)
        # if image_1.shape[0] != 3:
        #     image_1 = image_1.repeat(3, 1, 1)
        # if image_2.shape[0] != 3:
        #     image_2 = image_2.repeat(3, 1, 1)
        return (image_1, image_2) if not self.pose else (image_1, image_2, target_pose_2)

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.total_pairs_count


class InShopClothesRetrievalBenchmarkDataloader(DataLoader):

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                 image_transforms: Optional[transforms.Compose] = None, batch_size: int = 8, hq: bool = False, *args):
        """
        InShopClothesRetrievalBenchmarkDataloader class constructor.
        :param root: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param batch_size: the number of images batch
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        :param args: argument list for torch.utils.data.Dataloader constructor
        """
        _dataset = InShopClothesRetrievalBenchmarkDataset(root=root, image_transforms=image_transforms, hq=hq)
        super(InShopClothesRetrievalBenchmarkDataloader, self).__init__(dataset=_dataset, batch_size=batch_size, *args)


class InShopClothesRetrievalBenchmarkScraper:
    """
    InShopClothesRetrievalBenchmarkScraper Class:
    This class is used to scrape DeepFashion's In-shop Clothes Retrieval Benchmark images and extract image pairs.
    Forward Pass:
        Goes into every item folder (folders that are named item ID "id_<0-padded-8-digit-ID>") and creates a JSON file
        named "item_info.json" with information about images and cross-pose/scale pairs
    Backward Pass:
        Aggregates information on every category folder (e.g. "MEN/Shirts_Polos") of the items contained in the category
        and creates a JSON file "items_info.json". Recursive execution. After backward passes completion at the Img root
        directory there will be a huge JSON file "items_info.json" containing the image pairs for the whole dataset.
    When complete, in the Img root folder there will be a big JSON file containing the following aggregated information
        - id (int or str): "Img"
        - path (str): "/"
        - images_count_initial (int): initial number of benchmark's images
        - images_images (list): all usable image paths
        - images_count (int): usable number of benchmark's images
        - image_pairs (list):  [(pair1_image1, pair1_image2), (pair2_image1, pair2_image2) ...]
        - image_pairs_count (int): total number of image pairs in benchmark
        - image_groups (list):  ['group1_prefix_rel': [group1 images], ...]
        - image_groups_count (int): total number of image groups in benchmark
        (all paths are relative from Img folder)
    """

    def __init__(self, root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark', hq: bool = False):
        """
        InShopClothesRetrievalBenchmarkScraper class constructor.
        :param root: DeepFashion benchmark's root directory path
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        """
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
            if reason == 'IMAGE_SHORTAGE':
                return 2
        return 0

    def _forward_item(self, item_path_abs: str):
        """
        Processes single item. Creates item_info.json file inside $item_path_abs$ containing information about image
        groups and cross-pose/scale image pairs found in item directory.
        :param item_path_abs:
        :return:
        """
        item_images = [_f for _f in os.listdir(item_path_abs) if _f.endswith('.jpg')]
        item_image_pairs = []
        item_image_groups = group_by_prefix(item_images, '_')

        # Process each image group individually
        _to_delete_keys = []
        for group_prefix, group_images in item_image_groups.items():
            posable_images = [_image for _image in group_images
                              if self.__class__.is_posable(f'{group_prefix}_{_image}', item_path_abs)]
            if len(posable_images) < 2:
                # Remove group if yields no posable image pairs
                _to_delete_keys.append(group_prefix)
                continue

            group_image_pairs = get_pairs(posable_images, prefix=f'{group_prefix}_')
            item_image_pairs += group_image_pairs

        # Delete useless image groups
        for _k in _to_delete_keys:
            del item_image_groups[_k]

        # Stamp item if has no usable groups
        if 0 == len(item_image_groups):
            with open(f'{item_path_abs}/.skip', 'w') as skip_fp:
                skip_fp.write('\n'.join(['Reason: IMAGE_SHORTAGE', '']))
            self.logger.critical(f"Item {os.path.basename(item_path_abs)} should had been skipped...")
            return

        # Save item information in JSON file
        with open(f'{item_path_abs}/item_info.json', 'w') as json_fp:
            json.dump({
                'id': int(os.path.basename(item_path_abs).replace('id_', '')),
                'path': item_path_abs.replace(self.img_dir_path, ''),
                'images_count_initial': len(item_images),
                'images': join_lists(*[list(map(lambda _i: f'{group_prefix}_{_i}', group_images))
                                       for group_prefix, group_images in item_image_groups.items()]),
                'images_count': sum(len(group_images) for _, group_images in item_image_groups.items()),
                'image_groups': item_image_groups,
                'image_groups_count': len(item_image_groups.keys()),
                'image_pairs': item_image_pairs,
                'image_pairs_count': len(item_image_pairs),
            }, json_fp, indent=4)
        # self.logger.debug(f'{item_path_abs}/item_info.json: [DONE]')

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping DeepFashion ICRB images:
        Visits every item directory, process its images and saves image / pairs information a JSON file, item_info.json.
        """
        processed_ids = []
        useless_items = []
        processed_count = 0
        with tqdm(total=self.items_count, colour='yellow') as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                _json_files = [_f for _f in _files if _f.endswith('.json')]
                assert len(_dirs) * (len(_files) - len(_json_files)) == 0, \
                    "a dir should contain only sub-dirs of image and/or JSON files"

                if os.path.basename(_root).startswith('id_'):
                    item_type = self.__class__.get_item_type(_root)
                    if item_type in [-1, 1]:
                        continue
                    if item_type == 2:
                        # Duplicate item directory found. Skip processing.
                        # self.logger.warning(f'Found .skip with reason: DUPLICATE. Adding to useless...')
                        useless_items.append(os.path.basename(_root))
                    elif item_type == 0:
                        self._forward_item(_root)
                        processed_ids.append(os.path.basename(_root))
                        # self.logger.info(f'\tProcessed: {os.path.basename(_root)}')
                    else:
                        self.logger.critical('Really unexpected path occurred!')
                        assert False

                    processed_count += 1
                    progress_bar.update()

        if processed_count != self.items_count:
            self.logger.error(f'processed_count != total_count ({processed_count} != {self.items_count})')
            self.logger.error(str(list_diff(self.item_dirs, processed_ids)))

        if len(useless_items) > 0:
            self.logger.info(f'Found {len(useless_items)} useless items for pose change, i.e. items having 0 pairs')
            self.logger.info(str(useless_items))

    @staticmethod
    def _get_item_info(json_filepath_abs: str) -> dict:
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
            "images_count_initial": 4,
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
            "image_groups_count": 1,
            "image_pairs": [
                [
                    "01_1_front.jpg",
                    "01_2_side.jpg"
                ],
                ...
            ],
            "image_pairs_count": 6
        }
        """
        _keys = ['images_count_initial', 'images', 'images_count', 'image_groups', 'image_groups_count',
                 'image_pairs', 'image_pairs_count']
        _dict = dict((k, item_or_items_info_dict[k]) for k in _keys if k in item_or_items_info_dict)

        # Prefix images
        for _i, _name in enumerate(_dict['images']):
            _dict['images'][_i] = f'{prefix}{_name}'
        # _to_delete_keys = []
        for _k, _v in list(_dict['image_groups'].items()):
            _dict['image_groups'][f'{prefix}{_k}'] = _dict['image_groups'][_k].copy()
            del _dict['image_groups'][_k]
        # for _k in _to_delete_keys:
        #     del _dict['image_groups'][_k]
        for _i, _pair in enumerate(_dict['image_pairs']):
            _dict['image_pairs'][_i][0] = f'{prefix}{_pair[0]}'
            _dict['image_pairs'][_i][1] = f'{prefix}{_pair[1]}'

        return _dict

    def _backward_dir(self, depth: int, dir_abs: str, progress_bar: tqdm) -> None:
        _dirs = next(os.walk(dir_abs))[1]
        items_info_dict = {}
        for _dir in _dirs:
            _dir_abs = f'{dir_abs}/{_dir}'
            if os.path.exists(f'{_dir_abs}/.skip') or os.path.exists(f'{_dir_abs}/.duplicate'):
                continue

            _dir_info_dict = self.__class__._get_item_info(f'{_dir_abs}/item{"s" if depth < 2 else ""}_info.json')

            if not items_info_dict:
                # Initialize merged dict
                items_info_dict = _dir_info_dict.copy()
            else:
                # Merge _dir_info_dict with current items_info_dict
                items_info_dict['images_count_initial'] += _dir_info_dict['images_count_initial']
                items_info_dict['images'] += _dir_info_dict['images']
                items_info_dict['images_count'] += _dir_info_dict['images_count']
                # _merged_image_groups = {**items_info_dict['image_groups'], **_dir_info_dict['image_groups']}
                items_info_dict['image_groups'] = {**items_info_dict['image_groups'], **_dir_info_dict['image_groups']}
                items_info_dict['image_groups_count'] += _dir_info_dict['image_groups_count']
                items_info_dict['image_pairs'] += _dir_info_dict['image_pairs']
                items_info_dict['image_pairs_count'] += _dir_info_dict['image_pairs_count']

            progress_bar.update()

        items_info_dict['id'] = os.path.basename(dir_abs)
        items_info_dict['path'] = dir_abs.replace(self.img_dir_path, '')

        # Save item information in JSON file
        with open(f'{dir_abs}/items_info.json', 'w') as json_fp:
            json.dump(items_info_dict, json_fp, indent=4)

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
            2: (7982 - 12),  # 12 useless items
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

    def resolve_useless_items(self):
        """
        Method to perform useless items resolution:
        "Useless" items are the ones that yield zero image pairs. Upon finding such items we "stamp" them by adding a
        ".skip" file in item's directory.
        """
        with tqdm(total=self.items_count + 100, colour='yellow') as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                _json_files = [_f for _f in _files if _f.endswith('.json')]
                assert len(_dirs) * (len(_files) - len(_json_files)) == 0, \
                    "a dir should contain only sub-dirs of image and/or JSON files"

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
                    if 0 == len(item_image_groups):
                        with open(f'{_root}/.skip', 'w') as skip_fp:
                            skip_fp.write('\n'.join(['Reason: IMAGE_SHORTAGE', '']))
                        self.logger.debug(f"Item {os.path.basename(_root)} stamped")

                    progress_bar.update()

    def test_forward(self) -> bool:
        """
        Tests forward pass by re-visiting every directory and checking for information found in *_info.json files.
        :return: True if everything looks good, False if any error occurs
        """
        _result = True
        with tqdm(total=self.items_count + 100, colour='yellow') as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                if len(_files) > 0 and _files != ['items_info.json'] and _files != ['item_info.json']:
                    # Check if every POSABLE image has its corresponding pose image
                    _jpg_files = [_f for _f in _files if _f.endswith('.jpg') and self.__class__.is_posable(_f, _root)]
                    _jpg_count = len(_jpg_files)
                    _pose_files = [_f for _f in _files if _f.endswith('IUV.png')]
                    _pose_count = len(_pose_files)
                    if len(_jpg_files) != len(_pose_files):
                        if _jpg_count > _pose_count:
                            self.logger.critical(f'Found {_jpg_count} JPG files but only {_pose_count} PNG' +
                                                 f' (id_dir={_root})')
                        else:
                            self.logger.critical(f'Found {_pose_count} PNG files but only {_jpg_count} JPG' +
                                                 f' (id_dir={_root})')
                        _result = False
                    # Check items.info
                    if os.path.basename(_root).startswith('id_'):
                        if not os.path.exists(f'{_root}/item_info.json') and not (
                                os.path.exists(f'{_root}/.skip') or os.path.exists(f'{_root}/.duplicate')):
                            self.logger.critical(f'_root.startswith("id_"), len(_files) = {len(_files)} but no' +
                                                 f' "item_info.json" (id_dir={_root})')
                            _result = False
                        else:
                            # Check JSON
                            with open(f'{_root}/item_info.json') as fp:
                                item_info = json.load(fp)

                            if item_info['id'] != int(os.path.basename(_root).replace('id_', '')):
                                self.logger.critical(f'item_info.id mismatch ({item_info["id"]}, id_dir={_root})')
                                _result = False
                            elif item_info['path'] != _root.replace(self.img_dir_path, ''):
                                self.logger.critical(f'item_info.path mismatch ({item_info["path"]}, id_dir={_root})')
                                _result = False
                            elif item_info['images_count_initial'] != len([_f for _f in _files if _f.endswith('.jpg')]):
                                self.logger.critical(f'item_info.images_count_initial mismatch' +
                                                     f' ({item_info["images_count_initial"]}, id_dir={_root})')
                                _result = False
                if os.path.basename(_root).startswith('id_'):
                    if len(_files) == 0:
                        self.logger.critical(f'len(_files) = 0 in id root (id_dir={_root}')
                        _result = False
                    elif len(_files) == 1:
                        self.logger.critical(f'len(_files) = 1 in id root (id_dir={_root}')
                        _result = False
                    progress_bar.update()
        return _result

    def test_backward(self) -> bool:
        """
        Tests backward pass by re-visiting every non ID directory and checking for items_info.json file existence.
        :return: True if everything looks good, False if any error occurs
        """
        _result = True
        with tqdm(total=25, colour='yellow') as progress_bar:
            for _root, _dirs, _files in os.walk(self.img_dir_path):
                if not os.path.basename(_root).startswith('id_') and not os.path.exists(f'{_root}/items_info.json'):
                    self.logger.critical(f'_root.startswith("id_")=False but "items_info.json" not found' +
                                         f' (_root={_root}')
                    _result = False
                    progress_bar.update()
        return _result

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True, hq: bool = False) -> None:
        """
        Entry point of class.
        :param forward_pass: indicate if should run scraper's forward pass (create item_info.json files in item dirs)
        :param backward_pass: indicate if should run scraper's backward pass (recursively merge items JSON files)
                              Note: if $forward_pass$ is set to True, then $backward_pass$ is also set to True.
        :param hq: if True will process HQ versions of benchmark images (that live inside the Img/ImgHQ folder)
        """
        scraper = InShopClothesRetrievalBenchmarkScraper(hq=hq)
        scraper.logger.info(f'SCRAPE DIR = {scraper.img_dir_path}')
        scraper.logger.info('resolve_duplicate_items(): [STARTING...]')
        scraper.resolve_duplicate_items()
        scraper.logger.info(f'resolve_duplicate_items(): [DONE]')
        scraper.logger.info('resolve_useless_items(): [STARTING...]')
        scraper.resolve_useless_items()
        scraper.logger.info(f'resolve_useless_items(): [DONE]')
        if forward_pass:
            scraper.logger.info('forward(): [STARTING...]')
            scraper.forward()
            # assert scraper.test_forward()
            scraper.logger.info('forward(): [DONE]')
            backward_pass = True
        if backward_pass:
            scraper.logger.info('backward(): [STARTING...]')
            scraper.backward()
            # assert scraper.test_backward()
            scraper.logger.info('backward(): [DONE]')
        scraper.logger.info('[DONE]')


if __name__ == '__main__':
    InShopClothesRetrievalBenchmarkScraper.run(forward_pass=True, backward_pass=True, hq=False)
    # InShopClothesRetrievalBenchmarkScraper.run(forward_pass=True, backward_pass=True, hq=True)
