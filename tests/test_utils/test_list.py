import unittest

from utils.list import get_pairs
from utils.string import group_by_prefix


class TestListUtils(unittest.TestCase):

    def test_get_pairs(self):
        test_str_list = [
            'prefix1_suffix1',
            'prefix1_suffix1_1',
            'prefix1_suffix1_1_1',
            'prefix1_suffix2',

            'prefix2_suffix1',
            'prefix2_suffix2',
            'prefix2_suffix2_1',
            'prefix2_suffix2_1_1',

            'prefix3_suffix1',
            'prefix_3_suffix1',
        ]
        test_groups = group_by_prefix(test_str_list, separator='_')

        group_1_pairs = get_pairs(test_groups['prefix1'], exclude_same=True, forward_only=True, prefix='prefix1_')
        self.assertEqual(group_1_pairs, [
            ('prefix1_suffix1', 'prefix1_suffix1_1'),
            ('prefix1_suffix1', 'prefix1_suffix1_1_1'),
            ('prefix1_suffix1', 'prefix1_suffix2'),
            ('prefix1_suffix1_1', 'prefix1_suffix1_1_1'),
            ('prefix1_suffix1_1', 'prefix1_suffix2'),
            ('prefix1_suffix1_1_1', 'prefix1_suffix2'),
        ])

        group_2_pairs = get_pairs(test_groups['prefix2'], exclude_same=True, forward_only=True, prefix='prefix2_')
        self.assertEqual(group_2_pairs, [
            ('prefix2_suffix1', 'prefix2_suffix2'),
            ('prefix2_suffix1', 'prefix2_suffix2_1'),
            ('prefix2_suffix1', 'prefix2_suffix2_1_1'),
            ('prefix2_suffix2', 'prefix2_suffix2_1'),
            ('prefix2_suffix2', 'prefix2_suffix2_1_1'),
            ('prefix2_suffix2_1', 'prefix2_suffix2_1_1'),
        ])
