import json
import os
import sys
import unittest

import numpy as np
import torch
from IPython import get_ipython
from torchvision.transforms import transforms

from dataset.deep_fashion import ICRBCrossPoseDataset, ICRBDataset


class TestICRBDataset(unittest.TestCase):

    # noinspection DuplicatedCode
    def setUp(self) -> None:
        self.inside_colab = 'google.colab' in sys.modules or \
                            'google.colab' in str(get_ipython()) or \
                            'COLAB_GPU' in os.environ
        self.deep_fashion_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark'
        self.deep_fashion_root = f'{"/content" if self.inside_colab else ""}{self.deep_fashion_root}'
        self.deep_fashion_img_root = f'{self.deep_fashion_root}/Img'
        self.assertTrue(os.path.exists(f'{self.deep_fashion_img_root}/items_info.json'))
        with open(f'{self.deep_fashion_img_root}/items_info.json') as fp:
            self.items_info = json.load(fp)

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset = ICRBDataset(root=self.deep_fashion_root, image_transforms=self.transforms, hq=False)

    def test_len(self) -> None:
        self.assertEqual(self.items_info['images_count'], len(self.dataset))

    def test_getitem(self) -> None:
        for _index in np.random.choice(range(len(self.dataset)), 1000):
            image = self.dataset[_index]
            self.assertEqual(tuple(image.shape), (3, 256, 256), msg=f'image.shape={tuple(image.shape)}')


class TestICRBCrossPoseDataset(unittest.TestCase):

    # noinspection DuplicatedCode
    def setUp(self) -> None:
        self.inside_colab = 'google.colab' in sys.modules or \
                            'google.colab' in str(get_ipython()) or \
                            'COLAB_GPU' in os.environ
        self.deep_fashion_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark'
        self.deep_fashion_root = f'{"/content" if self.inside_colab else ""}{self.deep_fashion_root}'
        self.deep_fashion_img_root = f'{self.deep_fashion_root}/Img'
        self.assertTrue(os.path.exists(f'{self.deep_fashion_img_root}/items_info.json'))
        with open(f'{self.deep_fashion_img_root}/items_posable_info.json') as fp:
            self.items_posable_info = json.load(fp)

        self.target_shape = 128
        self.transforms = transforms.Compose([
            transforms.Resize(self.target_shape),
            transforms.CenterCrop(self.target_shape),
            transforms.ToTensor()
        ])
        self.dataset = ICRBCrossPoseDataset(root=self.deep_fashion_root, pose=False, image_transforms=self.transforms,
                                            hq=False)

    def test_len(self) -> None:
        self.assertEqual(self.items_posable_info['posable_image_pairs_count'], len(self.dataset) // 2)

    def test_index_to_paths(self) -> None:
        for _index in range(len(self.dataset) // 2):
            self.dataset.pose = False
            image_1_path, image_2_path = self.dataset.index_to_paths(_index)
            self.assertTrue(os.path.exists(image_1_path), msg=f'image_1_path={image_1_path} does NOT exist')
            self.assertTrue(os.path.exists(image_2_path), msg=f'image_2_path={image_2_path} does NOT exist')
            self.dataset.pose = True
            image_1_path, pose_1_path, image_2_path, pose_2_path = self.dataset.index_to_paths(_index)
            self.assertTrue(os.path.exists(image_1_path), msg=f'image_1_path={image_1_path} does NOT exist')
            self.assertTrue(os.path.exists(pose_1_path), msg=f'pose_1_path={pose_1_path} does NOT exist')
            self.assertTrue(os.path.exists(image_2_path), msg=f'image_2_path={image_2_path} does NOT exist')
            self.assertTrue(os.path.exists(pose_2_path), msg=f'pose_2_path={pose_2_path} does NOT exist')

    def test_getitem(self) -> None:
        for _index in np.random.choice(range(len(self.dataset)), 1000):
            self.dataset.pose = False
            image_1, image_2 = self.dataset[_index]
            self.assertEqual(tuple(image_1.shape), (3, self.target_shape, self.target_shape),
                             msg=f'image_1.shape={tuple(image_1.shape)}')
            self.assertEqual(tuple(image_2.shape), (3, self.target_shape, self.target_shape),
                             msg=f'image_2.shape={tuple(image_2.shape)}')
            # Check reverse pair
            if _index >= len(self.dataset) // 2:
                _index_rev = _index - len(self.dataset) // 2
            else:
                _index_rev = _index + len(self.dataset) // 2
            image_1_rev, image_2_rev = self.dataset[_index_rev]
            self.assertTrue(torch.all(image_1_rev.eq(image_2)))
            self.assertTrue(torch.all(image_2_rev.eq(image_1)))

            self.dataset.pose = True
            tensors_tuple = self.dataset[_index]
            self.assertEqual(len(tensors_tuple), 3)
            self.assertEqual(type(tensors_tuple[0]), torch.Tensor)
            self.assertEqual(tuple(tensors_tuple[0].shape), (3, self.target_shape, self.target_shape),
                             msg=f'tensors_tuple[0].shape={tuple(tensors_tuple[0].shape)}')
            self.assertEqual(tuple(tensors_tuple[1].shape), (3, self.target_shape, self.target_shape),
                             msg=f'tensors_tuple[0].shape={tuple(tensors_tuple[1].shape)}')
            self.assertEqual(tuple(tensors_tuple[2].shape), (3, self.target_shape, self.target_shape),
                             msg=f'tensors_tuple[2].shape={tuple(tensors_tuple[2].shape)}')
            # Check reverse pair
            image_1_rev, image_2_rev, target_2_rev = self.dataset[_index_rev]
            self.assertTrue(torch.all(image_2_rev.eq(tensors_tuple[0])))
            self.assertTrue(torch.all(image_1_rev.eq(tensors_tuple[1])))
            self.assertFalse(torch.all(target_2_rev.eq(tensors_tuple[2])))


class TestICRBScraper(unittest.TestCase):

    def setUp(self) -> None:
        self.inside_colab = 'google.colab' in sys.modules or \
                            'google.colab' in str(get_ipython()) or \
                            'COLAB_GPU' in os.environ
        self.path_prefix = '/content' if self.inside_colab else ''
        self.deep_fashion_root = self.path_prefix + '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark'
        self.deep_fashion_img_root = f'{self.deep_fashion_root}/Img'

    def test_forward(self) -> None:
        for _root, _, _files in os.walk(self.deep_fashion_img_root):
            if os.path.basename(_root).startswith('id_'):
                if '.duplicate' in _files:
                    self.assertFalse('.skip' in _files)
                    self.assertFalse('item_info.json' in _files)
                    self.assertFalse('item_posable_info.json' in _files)
                    with open(f'{_root}/.duplicate') as dup_fp:
                        dup_file_lines = dup_fp.read().splitlines()
                    self.assertTrue(2, len(dup_file_lines))
                    self.assertTrue(dup_file_lines[0].startswith('Moved To: '))
                    self.assertTrue(dup_file_lines[1].startswith('Reason: '))
                    moved_to_path = self.path_prefix + dup_file_lines[0].replace('Moved To: ', '')
                    self.assertTrue(os.path.exists(moved_to_path))
                    self.assertFalse(os.path.exists(f'{moved_to_path}/.duplicate'))
                elif '.skip' in _files:
                    self.assertFalse('.duplicate' in _files)
                    self.assertFalse('item_posable_info.json' in _files)
                    if len([_f for _f in _files if _f.endswith('.jpg')]) > 0:
                        self.assertTrue('item_info.json' in _files)
                    else:
                        self.assertFalse('item_info.json' in _files)
                else:
                    _image_files = [_f for _f in _files if _f.endswith('.jpg') or _f.endswith('.png')]
                    _non_image_files = [_f for _f in _files if not _f.endswith('.jpg') and not _f.endswith('.png')]
                    self.assertTrue(['item_info.json', 'item_posable_info.json'] == sorted(_non_image_files),
                                    msg=f'@{_root}, found files: {str(_files)}')
                    self.assertEqual(len(_files), len(_image_files) + len(_non_image_files),
                                     msg=f'@{_root}, lengths mismatch, files: {str(_files)}')

    def test_backward(self) -> None:
        for _root, _dirs, _files in os.walk(self.deep_fashion_img_root):
            if os.path.basename(_root).startswith('id_'):
                continue

            self.assertTrue(['items_info.json', 'items_posable_info.json'] == _files,
                            msg=f'@{_root}, found files: {str(_files)}')
