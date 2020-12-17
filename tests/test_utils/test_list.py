import unittest

from utils.list import get_pairs, list_diff, join_lists
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

    def test_list_diff(self):
        test_list1 = ['item1', 'item2', 'item4', 'item3']
        test_list2 = ['item1', 'item2', 'item', 'item3']
        diff = list_diff(test_list1, test_list2)
        self.assertEqual(['item4', 'item'], diff)
        self.assertEqual([], list_diff(diff, diff))
        self.assertEqual([], list_diff([], []))

    def test_join_lists(self):
        l1 = [{"id": "dict1"}, {"id": "dict2"}]
        l2 = [{"id": "dict3"}, {"id": "dict4"}]
        self.assertEqual([
            {"id": "dict1"}, {"id": "dict2"},
            {"id": "dict3"}, {"id": "dict4"}
        ], join_lists(l1, l2))
        self.assertEqual([
            {"id": "dict1"}, {"id": "dict2"},
            {"id": "dict1"}, {"id": "dict2"}
        ], join_lists(l1, l1))
        self.assertEqual([
            {"id": "dict1"}, {"id": "dict2"}
        ], join_lists(l1, []))
        self.assertEqual([], join_lists([], []))
