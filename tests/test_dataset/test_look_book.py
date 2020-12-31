import json
import os
import unittest

import numpy as np

from dataset.deep_fashion import ICRBDataset
from dataset.look_book import PixelDTDataset


# noinspection DuplicatedCode
class TestPixelDTDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.target_shape = 128
        self.transforms = PixelDTDataset.get_image_transforms(target_channels=3, target_shape=self.target_shape)
        self.dataset = PixelDTDataset(image_transforms=self.transforms)

        self.look_book_root = self.dataset.root
        self.look_book_img_root = f'{self.look_book_root}/Img'
        self.assertTrue(os.path.exists(f'{self.look_book_img_root}/items_dt_info.json'))
        with open(f'{self.look_book_img_root}/items_dt_info.json') as fp:
            self.items_dt_info = json.load(fp)

    def test_len(self) -> None:
        self.assertEqual(self.items_dt_info['dt_image_pairs_count'], len(self.dataset))

    def test_index_to_paths(self) -> None:
        for _index in range(len(self.dataset)):
            image_1_path, image_2_path = self.dataset.index_to_paths(_index)
            self.assertTrue(os.path.exists(image_1_path), msg=f'image_1_path={image_1_path} does NOT exist')
            self.assertTrue(os.path.exists(image_2_path), msg=f'image_2_path={image_2_path} does NOT exist')

    def test_getitem(self) -> None:
        for _index in np.random.choice(range(len(self.dataset)), 1000):
            image_1, image_2 = self.dataset[_index]
            self.assertEqual(tuple(image_1.shape), (3, self.target_shape, self.target_shape),
                             msg=f'image_1.shape={tuple(image_1.shape)}')
            self.assertEqual(tuple(image_2.shape), (3, self.target_shape, self.target_shape),
                             msg=f'image_2.shape={tuple(image_2.shape)}')


# noinspection DuplicatedCode
class TestPixelDTScraper(unittest.TestCase):

    def setUp(self) -> None:
        self.path_prefix = ICRBDataset.get_root_prefix()
        self.look_book_root = f'{self.path_prefix}/data/Datasets/LookBook'
        self.look_book_img_root = f'{self.look_book_root}/Img'

    def test_forward(self) -> None:
        for _root, _, _files in os.walk(self.look_book_img_root):
            if os.path.basename(_root).startswith('id_'):
                # Check if .src file is present in items from DeepFashion
                pid = int(os.path.basename(_root).replace('id_', ''))
                self.assertEqual(pid >= 8726, '.src' in _files)
                # Check image files
                _image_files = [_f for _f in _files if _f.endswith('.jpg')]
                self.assertTrue('flat.jpg' in _image_files)
                self.assertGreaterEqual(len(_image_files), 2)
                # Check non-image files
                _non_image_files = [_f for _f in _files if not _f.endswith('.jpg')]
                self.assertTrue('item_dt_info.json' in _non_image_files,
                                msg=f'@{_root}, found non image files: {str(_non_image_files)}')
                self.assertEqual(len(_files), len(_image_files) + len(_non_image_files),
                                 msg=f'@{_root}, lengths mismatch, files: {str(_files)}')

    def test_backward(self) -> None:
        for _root, _dirs, _files in os.walk(self.look_book_img_root):
            if os.path.basename(_root).startswith('id_'):
                continue

            self.assertTrue(['items_dt_info.json'] == _files, msg=f'@ {_root}, found files: {str(_files)}')
