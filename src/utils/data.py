import json
import os
import os.path
import prettytable
from typing import Optional, List

from utils.pytorch import to_human_readable


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


def deep_fashion_icrb_info(deep_fashion_root_dir: str, hq: bool = False, return_dict: bool = False,
                           update_json: bool = False, use_json: bool = False,
                           print_dict: bool = True) -> Optional[List[dict]]:
    """
    Display DeepFashion In-shop Clothes Retrieval Benchmark (ICRB) information.
    e.g call: deep_fashion_icrb_info(deep_fashion_root_dir='/data/Datasets/DeepFashion', use_json=True, print_dict=True)
    :param deep_fashion_root_dir: the root dir of DeepFashion dataset
    :param hq: use HQ images of benchmark instead of the 256x256 images
    :param return_dict: if True returns calculated dictionary with folder/file info
    :param print_dict: if True prints list with PrettyTable lib
    :param use_json: if True fetches info from JSON/saves info to JSON
    :param update_json: if True and $use_json$ is True and json file exists, it deletes file and re-creates it
    """
    img_dir = f'{deep_fashion_root_dir}/In-shop Clothes Retrieval Benchmark/Img{"HQ" if hq else ""}'
    json_filepath = f'{img_dir}/img{"hq" if hq else ""}_info.json'

    def _print_dict(_dict: List[dict]):
        _dict = [{'path': '', 'category': '', 'id_dirs_count': '', 'files_count': ''}] + _dict
        table = prettytable.from_json(json.dumps(info_dict))
        table.field_names = ["path", "category", "id_dirs_count", "files_count"]
        print(table)

    if use_json and os.path.exists(json_filepath):
        if not update_json:
            with open(json_filepath) as json_file:
                info_dict = json.load(json_file)
                if print_dict:
                    _print_dict(info_dict)
            return info_dict if return_dict else None
        else:
            os.remove(json_filepath)

    info_dict = []
    total_dirs_count = total_files_count = 0
    for root, dirs, files in os.walk(img_dir):
        if os.path.basename(root).startswith('id_'):
            continue

        files_count = len(files)
        dirs_count = len(dirs)
        last_category = root.replace(img_dir, '').lower().replace('_', '-').lstrip('/')
        is_parent_of_id_dirs = dirs_count > 0 and dirs[0].startswith('id_')
        if is_parent_of_id_dirs and files_count > 0:
            raise ValueError
        if is_parent_of_id_dirs:
            id_dirs_count = count_dirs(root, recursive=True)
            id_files_count = count_files(root, recursive=True)
            info_dict.append({
                'path': root.replace(img_dir, ''),
                'category': last_category,
                'id_dirs_count': to_human_readable(id_dirs_count),
                'files_count': to_human_readable(id_files_count),
            })
            total_dirs_count += id_dirs_count
            total_files_count += id_files_count

    # Append total files count
    info_dict.append({
        'path': '/',
        'category': '[*]',
        'id_dirs_count': f'{to_human_readable(total_dirs_count)} ({total_dirs_count})',
        'files_count': f'{to_human_readable(total_files_count)} ({total_files_count})',
    })

    if use_json:
        with open(json_filepath, 'w') as json_file:
            json.dump(info_dict, fp=json_file, indent=4)

    if print_dict:
        _print_dict(info_dict)

    return info_dict if return_dict else None
