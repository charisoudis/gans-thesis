import itertools
from typing import List, Optional


def get_pairs(input_list: list, exclude_same: bool = True, forward_only: bool = True, prefix: Optional[str] = None)\
        -> List[tuple]:
    """
    Pair-ify :attr:`input_string` be returning a list of tuples containing pairs of items from list.
    :param input_list: the input list
    :param exclude_same: if True, will exclude (item, item) tuples from paired output list
    :param forward_only: if True, will exclude associative pairs
    :param prefix: prefix applied
    :return: a list of pairs (tuples)
    """
    paired_list = []
    for item1 in input_list:
        for item2 in input_list:
            if item1 == item2 and exclude_same:
                continue

            item1_prefixed = (prefix or '') + item1
            item2_prefixed = (prefix or '') + item2
            if (item2_prefixed, item1_prefixed) in paired_list and forward_only:
                continue
            paired_list.append((item1_prefixed, item2_prefixed))
    return paired_list


def list_diff(l1: list, l2: list) -> List:
    """
    Get difference between two list objects.
    :param l1: 1st input list
    :param l2: 2nd input list
    :return: a list containing items that are present in the union of the two input lists but not present in either of
             them
    """
    return [_i for _i in l1 + l2 if _i not in l1 or _i not in l2]


def join_lists(*lists: list, map_fn: Optional = None) -> list:
    """
    Joins a list of list objects by "appending" all together.
    :param lists: variable length argument list containing input lists (ex. call: join_lists(list1, list2, list3))
    :param map_fn: if provided then every list in $lists$ is passed via $map_fn$ element-by-element (see map())
    :return: the resulting joint list
    """
    if map_fn is not None:
        lists = [list(map(map_fn, _l)) for _l in lists]
    return list(itertools.chain(*lists))
