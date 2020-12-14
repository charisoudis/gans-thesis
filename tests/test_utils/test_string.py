import unittest

from utils.string import group_by_prefix


class TestStringUtils(unittest.TestCase):

    def test_group_by_prefix(self):
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
        self.assertEqual(dict, type(test_groups))
        self.assertEqual(('prefix1', 'prefix2', 'prefix3', 'prefix'), tuple(test_groups.keys()))
        self.assertEqual(list, type(test_groups['prefix1']))
        self.assertEqual(4, len(test_groups['prefix1']))
        self.assertEqual(('suffix1', 'suffix1_1', 'suffix1_1_1', 'suffix2'), tuple(test_groups['prefix1']))
        self.assertEqual(('suffix1', 'suffix2', 'suffix2_1', 'suffix2_1_1'), tuple(test_groups['prefix2']))
        self.assertEqual(('suffix1',), tuple(test_groups['prefix3']))
        self.assertEqual(('3_suffix1',), tuple(test_groups['prefix']))
