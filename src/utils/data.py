import atexit
import io
import json
import os
import os.path
from typing import Optional, List, Union, Sized

import prettytable
import torch
from PIL import Image
# noinspection PyProtectedMember
from torch.utils.data import Sampler

from utils.string import to_human_readable


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


class ResumableRandomSampler(Sampler):
    """
    ResumableRandomSampler Class:
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    Original source: https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5
    """

    def __init__(self, data_source: Sized, shuffle: bool = True, seed: int = 42):
        """
        ResumableRandomSampler class constructor.
        generator (Generator): Generator used in sampling.
        :param data_source: torch.utils.data.Dataset or generally typings.Sized object of the dataset to sample from
        :param seed: generator manual seed parameter
        """
        super(ResumableRandomSampler, self).__init__(data_source=data_source)

        self.n_samples = len(data_source)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.shuffle = shuffle
        if self.shuffle:
            self.perm_index = None
            self.perm = None
            self.reshuffle()
        else:
            self.perm_index = 0
            self.perm = range(0, self.n_samples)

    def reshuffle(self) -> None:
        self.perm_index = 0
        self.perm = list(torch.randperm(self.n_samples, generator=self.generator).numpy())

    def __iter__(self):
        # If reached the end of dataset, reshuffle
        if self.perm_index >= len(self.perm):
            if self.shuffle:
                self.reshuffle()
            else:
                self.perm_index = 0

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.n_samples

    def get_state(self) -> dict:
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state: dict) -> None:
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


def pltfig_to_pil(_plt):
    buf = io.BytesIO()
    _plt.savefig(buf, format='jpg')
    buf.seek(0)
    im = Image.open(buf)
    atexit.register(buf.close)
    return im


def squarify_img(img: Union[str, Image.Image], target_shape: Optional[int] = None,
                 bg_color: Union[str, float, int] = 'white'):
    """
    Converts PIL image to square by expanding its smaller dimension and painting the background according to given
    :attr:`bg_color`.
    Source: https://github.com/nkmk/python-tools/blob/0178324f04579b8bab636136eb14776702ccf554/tool/lib/imagelib.py
    :param img: the input image as a PIL.Image object or a filepath string
    :param (optional) target_shape: if not None, the image will be resized to the given shape
                                    (width=height=:attr:`target_shape`)
    :param bg_color: background color as int (0, 255) or float (0, 1) or string (e.g. 'white')
    :return: a PIL.Image object containing the resulting square image
    """
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        pil_img = Image.open(img)
    width, height = pil_img.size
    # Squarify
    if width == height:
        result = pil_img
    elif width > height:
        result = Image.new(pil_img.mode, size=(width, width), color=bg_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), color=bg_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    # Resize
    if target_shape:
        result = result.resize(size=(target_shape, target_shape), resample=Image.BICUBIC)
    return result


def unzip_file(zip_filepath: str) -> bool:
    """
    Unzips a zip file at given :attr:`zip_filepath` using `unzip` lib & shell command.
    :param zip_filepath: the absolute path to the .zip file
    :return: a `bool` object set to True if shell command return 0, False otherwise
    """
    return True if 0 == os.system(f'unzip -q "{zip_filepath}" -d ' +
                                  f'"{zip_filepath.replace("/" + os.path.basename(zip_filepath), "")}"') \
        else False
