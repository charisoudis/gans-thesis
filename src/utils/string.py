from typing import List

import humanize


def group_by_prefix(str_list: List[str], separator: str = '_') -> dict:
    """
    Groups a list of strings by the common prefixes found in the strings.
    :param str_list: list of strings that will be grouped by their prefixes
    :param separator: prefix separator, splits string in two parts: before and after the 1st appearance of the separator
    :return: dict in the form {'prefix1': [suffix1, suffix2, ...], 'prefix2': [suffix1, suffix2, ...]}
    """
    strings_by_prefix = {}
    for s in str_list:
        prefix, suffix = map(str.strip, s.split(sep=separator, maxsplit=1))
        group = strings_by_prefix.setdefault(prefix, [])
        group.append(suffix)
    return strings_by_prefix


def to_human_readable(number: int, size_format: str = '%.1f', return_number: bool = False) -> str:
    """
    Convert input number to a human-readable string (e.g. 15120 --> 15K)
    :param number: the input integer
    :param size_format: format argument of humanize.naturalsize()
    :param return_number: set to True to return input number after human-readable and inside parentheses
    :return: human-readable formatted string
    """
    string = humanize.naturalsize(number, format=size_format)
    string = string.replace('.0', '').replace('Byte', '').replace('kB', 'K').rstrip('Bs').replace(' ', '')
    string = string.replace('G', 'B')   # billions
    return string + (f' ({number})' if return_number else '')
