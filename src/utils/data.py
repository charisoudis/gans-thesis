import json
import os
import os.path
import random
from typing import Optional, List

import numpy as np
import prettytable
import torch

from utils.ifaces import Reproducible
from utils.string import to_human_readable


# noinspection PyProtectedMember


def count_dirs(path: str, recursive: bool = False) -> int:
    """
    Get the number of directories under the given path.
    :param path: the root path to start searching for directories
    :param recursive: if True goes into every directory and counts sub-directories recursively
    :return: the total number of directories (and sub-directories if $recursive$ is set) under given $path$
    """
    return sum(len(dirs) for _, dirs, _ in os.walk(path)) if recursive else len(next(os.walk(path))[1])


def count_files(path: str, recursive: bool = False) -> int:
    """
    Get the number of files under the given path.
    :param path: the root path to start searching for files
    :param recursive: if True goes into every directory and counts files in sub-directories in a recursive manner
    :return: the total number of files in $path$ (and sub-directories of $path$ if $recursive$ is set)
    """
    return sum(len(files) for _, _, files in os.walk(path)) if recursive else len(next(os.walk(path))[2])


def deep_fashion_icrb_info(root: str = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark',
                           hq: bool = False, return_dict: bool = False, print_table: bool = True) \
        -> Optional[List[dict]]:
    """
    Display DeepFashion In-shop Clothes Retrieval Benchmark (ICRB) information.
    e.g call: deep_fashion_icrb_info(deep_fashion_root_dir='/data/Datasets/DeepFashion', use_json=True, print_dict=True)
    :param root: the root dir of DeepFashion In-shop Clothes Retrieval Benchmark dataset
    :param hq: use HQ images of benchmark instead of the 256x256 images
    :param return_dict: if True returns calculated dictionary with folder/file info
    :param print_table: if True prints list with PrettyTable lib
    """
    img_dir = f'{root}/Img{"HQ" if hq else ""}'
    table = prettytable.PrettyTable(["path", "category", "images_count", "image_groups_count", "image_pairs_count"])
    ic = igc = ipc = 0
    for _root, _dirs, _files in os.walk(img_dir):
        if _root.endswith('MEN') or _root.endswith('/Img') or os.path.basename(_root).startswith('id_'):
            continue
        items_info_json_filepath = f'{_root}/items_info.json'
        if not os.path.exists(items_info_json_filepath):
            raise FileNotFoundError(f'Info file not found (tried {_root}/items_info.json)')
        with open(items_info_json_filepath, 'r') as json_fp:
            items_info = json.load(json_fp)

        category = items_info['path'].lower().replace('_', '-')
        table.add_row((items_info['path'], category, to_human_readable(items_info['images_count']),
                       to_human_readable(items_info['image_groups_count']),
                       to_human_readable(items_info['image_pairs_count'])))
        ic += items_info['images_count']
        igc += items_info['image_groups_count']
        ipc += items_info['image_pairs_count']
    table.add_row(('/', '[*]', to_human_readable(ic, return_number=True), to_human_readable(igc, return_number=True),
                   to_human_readable(ipc, return_number=True)))
    if print_table:
        print(table)
    return json.loads(table.get_json_string()) if return_dict else None


class ManualSeedReproducible(Reproducible):

    @staticmethod
    def manual_seed(seed: int) -> int:
        if Reproducible.is_seeded():
            return Reproducible._seed
        # Set seeder value
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        Reproducible._seed = seed
        return seed


def unzip_file(zip_filepath: str) -> bool:
    """
    Unzips a zip file at given :attr:`zip_filepath` using `unzip` lib & shell command.
    :param zip_filepath: the absolute path to the .zip file
    :return: a `bool` object set to True if shell command return 0, False otherwise
    """
    return True if 0 == os.system(f'unzip -q "{zip_filepath}" -d ' +
                                  f'"{zip_filepath.replace("/" + os.path.basename(zip_filepath), "")}"') \
        else False


def unnanify(y: np.ndarray) -> np.ndarray:
    """
    Remove NaNs from np array.
    (source: https://stackoverflow.com/a/6520696/13634700)
    :param (np.ndarray) y: input 1d array
    :return: an 1d array as np.ndarray object
    """
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y_out = y.copy()
    y_out[nans] = np.interp(x(nans), x(~nans), y_out[~nans])
    return y_out
