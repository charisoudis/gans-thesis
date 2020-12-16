import json
import os
import unittest

import torch
from torchvision.transforms import transforms

from dataset.deep_fashion import InShopClothesRetrievalBenchmarkDataset


class TestInShopClothesRetrievalBenchmarkDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.deep_fashion_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark'
        self.deep_fashion_img_root = f'{self.deep_fashion_root}/Img'
        self.assertTrue(os.path.exists(f'{self.deep_fashion_img_root}/items_info.json'))
        with open(f'{self.deep_fashion_img_root}/items_info.json') as fp:
            self.items_info = json.load(fp)

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset = InShopClothesRetrievalBenchmarkDataset(root=self.deep_fashion_root, pose=False,
                                                              image_transforms=self.transforms, hq=False)

    def test_len(self):
        self.assertEqual(self.items_info['image_pairs_count'], len(self.dataset) // 2)

    def test_index_to_paths(self):
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

    def test_getitem(self):
        for _index in range(len(self.dataset) // 2):
            self.dataset.pose = False
            image_1, image_2 = self.dataset[_index]
            self.assertEqual(tuple(image_1.shape), (3, 256, 256), msg=f'image_1.shape={tuple(image_1.shape)}')
            self.assertEqual(tuple(image_2.shape), (3, 256, 256), msg=f'image_2.shape={tuple(image_2.shape)}')
            self.dataset.pose = True
            tensors_tuple = self.dataset[_index]
            self.assertEqual(len(tensors_tuple), 3)
            self.assertEqual(type(tensors_tuple[0]), torch.Tensor)
            self.assertEqual(tuple(tensors_tuple[0].shape), (3, 256, 256),
                             msg=f'tensors_tuple[0].shape={tuple(tensors_tuple[0].shape)}')
            self.assertEqual(tuple(tensors_tuple[1].shape), (3, 256, 256),
                             msg=f'tensors_tuple[0].shape={tuple(tensors_tuple[1].shape)}')
            self.assertEqual(tuple(tensors_tuple[2].shape), (3, 256, 256),
                             msg=f'tensors_tuple[2].shape={tuple(tensors_tuple[2].shape)}')
