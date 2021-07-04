import random
import string
from typing import List, Union, Optional

import humanize


def group_by_prefix(str_list: Union[List[str], List[dict]], separator: str = '_',
                    dict_key: Optional[str] = None) -> dict:
    """
    Groups a list of strings by the common prefixes found in the strings.
    :param str_list: list of strings that will be grouped by their prefixes
    :param separator: prefix separator, splits string in two parts: before and after the 1st appearance of the
                      separator (defaults to "_")
    :param (optional) dict_key: if str_list contains dictionaries, this will be used to extract the key to separate
                                strings upon
    :return: dict in the form {'prefix1': [suffix1, suffix2, ...], 'prefix2': [suffix1, suffix2, ...]}
    """
    strings_by_prefix = {}
    for s in str_list:
        _key_to_split = s[dict_key] if isinstance(s, dict) else s
        prefix, suffix = map(str.strip, _key_to_split.split(sep=separator, maxsplit=1))
        group = strings_by_prefix.setdefault(prefix, [])
        group.append(suffix if type(s) == str else s)
    return strings_by_prefix


def to_human_readable(number: int, size_format: str = '%.1f', return_number: bool = False) -> str:
    """
    Convert input number to a human-readable string (e.g. 15120 --> 15K)
    :param number: the input integer
    :param size_format: format argument of humanize.naturalsize()
    :param return_number: set to True to return input number after human-readable and inside parentheses
    :return: human-readable formatted string
    """
    num_string = humanize.naturalsize(number, format=size_format)
    num_string = num_string.replace('.0', '').replace('Byte', '').replace('kB', 'K').rstrip('Bs').replace(' ', '')
    num_string = num_string.replace('G', 'B')  # billions
    return num_string + (f' ({number})' if return_number else '')


def get_random_string(length: int) -> str:
    """
    Get a random string containing ASCII alphanumerical characters.
    :param length: the length of generated string
    :return: a str object with length equal to :attr:`length` containing random characters.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
