import os
import shutil
import unittest

from datasets.samplers import ResumableRandomSampler
from utils.data import count_dirs, count_files


class TestDataUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.pwd = os.getcwd()
        self.test_dir = f'{self.pwd}/.TestDataUtils_test_dir'

        # Create 2 test dirs, the 1st with 2 sub-dirs and the other with 3 sub-dirs
        os.mkdir(f'{self.test_dir}')
        os.mkdir(f'{self.test_dir}/dir_1')
        os.mkdir(f'{self.test_dir}/dir_1/dir_1_1')
        os.mkdir(f'{self.test_dir}/dir_1/dir_1_2')
        os.mkdir(f'{self.test_dir}/dir_2')
        os.mkdir(f'{self.test_dir}/dir_2/dir_2_1')
        os.mkdir(f'{self.test_dir}/dir_2/dir_2_2')
        os.mkdir(f'{self.test_dir}/dir_2/dir_2_3')

        # The 1st dir will have 1 file inside the root and 2 files inside each of its sub-dirs
        dir_1_file_paths = (
            'dir_1/file_1.txt',
            'dir_1/dir_1_1/file_1.txt',
            'dir_1/dir_1_1/file_2.txt',
            'dir_1/dir_1_2/file_1.txt',
            'dir_1/dir_1_2/file_2.txt',
        )
        for _filepath in dir_1_file_paths:
            open(f'{self.test_dir}/{_filepath}', 'w').close()
        # The 2nd dir will have 2 files inside the root and (1,3,2) files inside each of its sub-dirs
        dir_2_file_paths = (
            'dir_2/file_1.txt',
            'dir_2/file_2.txt',
            'dir_2/dir_2_1/file_1.txt',
            'dir_2/dir_2_2/file_1.txt',
            'dir_2/dir_2_2/file_2.txt',
            'dir_2/dir_2_2/file_3.txt',
            'dir_2/dir_2_3/file_1.txt',
            'dir_2/dir_2_3/file_2.txt',
        )
        for _filepath in dir_2_file_paths:
            open(f'{self.test_dir}/{_filepath}', 'w').close()

    def test_count_dirs(self):
        self.assertEqual(2, count_dirs(self.test_dir))
        self.assertEqual(7, count_dirs(self.test_dir, recursive=True))
        self.assertEqual(2, count_dirs(f'{self.test_dir}/dir_1'))
        self.assertEqual(3, count_dirs(f'{self.test_dir}/dir_2'))
        self.assertEqual(0, count_dirs(f'{self.test_dir}/dir_1/dir_1_1'))
        self.assertEqual(0, count_dirs(f'{self.test_dir}/dir_1/dir_1_2'))
        self.assertEqual(0, count_dirs(f'{self.test_dir}/dir_2/dir_2_1'))
        self.assertEqual(0, count_dirs(f'{self.test_dir}/dir_2/dir_2_2'))
        self.assertEqual(0, count_dirs(f'{self.test_dir}/dir_2/dir_2_3'))

    def test_count_files(self):
        self.assertEqual(0, count_files(self.test_dir))
        self.assertEqual(13, count_files(self.test_dir, recursive=True))
        self.assertEqual(1, count_files(f'{self.test_dir}/dir_1'))
        self.assertEqual(5, count_files(f'{self.test_dir}/dir_1', recursive=True))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_1/dir_1_1'))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_1/dir_1_1', recursive=True))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_1/dir_1_2'))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_1/dir_1_2', recursive=True))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_2'))
        self.assertEqual(8, count_files(f'{self.test_dir}/dir_2', recursive=True))
        self.assertEqual(1, count_files(f'{self.test_dir}/dir_2/dir_2_1'))
        self.assertEqual(1, count_files(f'{self.test_dir}/dir_2/dir_2_1', recursive=True))
        self.assertEqual(3, count_files(f'{self.test_dir}/dir_2/dir_2_2'))
        self.assertEqual(3, count_files(f'{self.test_dir}/dir_2/dir_2_2', recursive=True))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_2/dir_2_3'))
        self.assertEqual(2, count_files(f'{self.test_dir}/dir_2/dir_2_3', recursive=True))

    def test_ResumableRandomSampler(self) -> None:
        sized_source = range(0, 1000)
        sampler = ResumableRandomSampler(data_source=sized_source, shuffle=False)
        self.assertListEqual([i for i in sized_source], [s_i for s_i in sampler])

        sized_source = range(0, 10)
        test_indices_with_seed_42 = [2, 6, 1, 8, 4, 5, 0, 9, 3, 7]
        sampler2 = ResumableRandomSampler(data_source=sized_source, shuffle=True, seed=42)
        self.assertListEqual(test_indices_with_seed_42, [s_i for s_i in sampler2])

        test_indices = [i for i in range(0, 10)] + [i for i in range(0, 10)]
        sampler3 = ResumableRandomSampler(data_source=range(0, 10), shuffle=False)
        self.assertListEqual(test_indices, [next(iter(sampler3)) for _ in range(20)])

        sized_source = range(0, 10)
        test_indices_with_seed_42 = [2, 6, 1, 8, 4, 5, 0, 9, 3, 7, 2, 6, 1, 8, 4, 5, 0, 9, 3, 7]
        sampler4 = ResumableRandomSampler(data_source=sized_source, shuffle=True, seed=42)
        sampler_indices = [next(iter(sampler4)) for _ in range(20)]
        self.assertEqual(len(test_indices_with_seed_42), len(sampler_indices))
        self.assertListEqual(test_indices_with_seed_42[:10], sampler_indices[:10])

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
