import atexit
import io
import json
import os
import shutil
from typing import Union, Optional

import matplotlib
import matplotlib.font_manager
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchvision.transforms import transforms

from utils.filesystems.gdrive.remote import GDriveFolder


def create_img_grid(images: torch.Tensor, ncols: int, nrows: Optional[int] = None, border: int = 2,
                    black: float = 0.5, gen_transforms: Optional[transforms.Compose] = None) -> torch.Tensor:
    """

    :param (torch.Tensor) images: torch.Tensor object of shape N x (CxHxW) (similar to training tensors)
    :param (int) ncols:
    :param (int) nrows:
    :param (int) border:
    :param (int) black:
    :param (optional) gen_transforms:
    :return:
    """

    # Inverse generator transforms
    if gen_transforms is not None:
        from utils.pytorch import invert_transforms
        gen_transforms_inv = invert_transforms(gen_transforms)
    else:
        from utils.pytorch import ToTensorOrPass
        gen_transforms_inv = ToTensorOrPass()

    # Check nrows
    if nrows is None:
        nrows = int(images.shape[0] / ncols)
    assert nrows * ncols == images.shape[0], 'nrows * ncols must be equal to the total number of images'

    # Create a single (grouped) image for each row
    row_images = []
    for r in range(nrows):
        _rlist = []
        _rheight = None
        for c in range(ncols):
            # Apply inverse image transforms to given images
            image = gen_transforms_inv(images[r * ncols + c]).float()
            if _rheight is None:
                _rheight = image.shape[1]
            _rlist.append(black * torch.ones(3, _rheight, border).float())  # |
            _rlist.append(image)  # |□
        _rlist.append(black * torch.ones(3, _rheight, border).float())  # |□|
        row_images.append(torch.cat(_rlist, dim=2).cpu())

    # Join row-images to form the final image
    _list = []
    for ri in row_images:
        _list.append(black * torch.ones(3, border, ri.shape[2]).float())  # ___
        _list.append(ri)  # |□|
        _list.append(black * torch.ones(3, border, ri.shape[2]).float())  # ---
        _list.append(1.0 * torch.ones(3, 4 * border, ri.shape[2]).float())  # (gap)
    return torch.cat(_list[:-1], dim=1).cpu()


def ensure_matplotlib_fonts_exist(groot: GDriveFolder, force_rebuild: bool = False) -> bool:
    """
    Downloads all TTF files from Google Drive's "Fonts" folder and places them in the directory `matplotlib` expects to
    find its .ttf font files.
    :param (GDriveFolder) groot: the parent of "Fonts" folder in Google Drive
    :param (bool) force_rebuild: set to True to forcefully rebuild fonts cache in matplotlib
    :return: a `bool` object set to `True` if fonts rebuilding was performed, `False` otherwise
    """
    # Get fonts gfolder
    fonts_gfolder = groot if 'Fonts' == groot.name else \
        groot.subfolder_by_name('Fonts')
    # Download all fonts from Google Drive
    fonts_gfolder.download(recursive=True, in_parallel=False, show_progress=True, unzip_after=False)
    # Define the matplotlib ttf font dir (destination dir)
    matplotlib_ttf_path = matplotlib.matplotlib_fname().replace('matplotlibrc', 'fonts/ttf')
    assert os.path.exists(matplotlib_ttf_path) and os.path.isdir(matplotlib_ttf_path)
    # Visit all subfolders and copy .ttf files to matplotlib fonts dir
    new_ttf_files = []
    for sf in fonts_gfolder.subfolders:
        sf_fonts_folder = f'/usr/share/fonts/truetype/{sf.name.replace(" ", "").lower()}'

        # Copy only JetBrains Mono font
        if not sf_fonts_folder.endswith('jetbrainsmono'):
            continue
        os.system(f'mkdir -p {sf_fonts_folder}')
        for f in sf.files:
            if not f.name.endswith('ttf'):
                continue
            # Copy file to matplotlib folder
            if not os.path.exists(f'{matplotlib_ttf_path}/{f.name}'):
                new_ttf_files.append(shutil.copy(f.path, matplotlib_ttf_path))
            # Copy to system fonts folder
            if not os.path.exists(f'{sf_fonts_folder}/{f.name}'):
                shutil.copy(f.path, sf_fonts_folder)
    # Inform and rebuild fonts cache
    rebuild = force_rebuild
    if len(new_ttf_files) > 0:
        print('Font files copied:')
        print(json.dumps(new_ttf_files, indent=4))
        rebuild = True
    if rebuild:
        # Rebuild system font cache
        os.system('fc-cache -fv')
        # Rebuild matplotlib font cache
        os.system('rm ~/.cache/matplotlib -rf')
        os.system('mkdir -p ~/.cache/matplotlib')
        # noinspection PyProtectedMember
        matplotlib.font_manager._rebuild()
    return rebuild


def plot_grid(grid: torch.Tensor or np.ndarray, figsize=tuple, footnote_l: Optional[str] = None,
              footnote_r: Optional[str] = None) -> Image:
    """
    Plots and image grid (created with `utils.plot.create_img_grid`)
    :param (torch.Tensor or numpy.ndarray) grid: a torch.Tensor or NumPy array object holding the image grid
    :param (tuple) figsize: plt.figure's `figsize` parameter as a tuple object (width, height). A wise choice could be
                            (ncols, nrows) as those were given to create_img_grid function.
    :param (optional) footnote_l: left-aligned footnote (is printed at the bottom of the plot)
    :param (optional) footnote_r: right-aligned footnote (is printed at the bottom of the plot -
                                  independent positioning of left footnote)
    :return: a `PIL.Image.Image` instance capturing the currently-shown matplotlib figure
    """
    # Set matplotlib params
    matplotlib.rcParams["font.family"] = 'JetBrains Mono'
    # Create a new figure
    plt.figure(figsize=figsize, dpi=300, frameon=False, clear=True)
    # Remove annoying axes
    plt.axis('off')
    # Create image and return
    if footnote_l:
        plt.suptitle(footnote_l, y=0.03, fontsize=4, fontweight='light', horizontalalignment='left', x=0.001)
    print(type(grid))
    plt.imshow(grid.permute(1, 2, 0) if isinstance(grid, torch.Tensor) else np.transpose(grid, (1, 2, 0)))
    # Show the right footnote
    fig: matplotlib.figure.Figure
    fig = plt.gcf()
    if footnote_r:
        fig.text(x=1, y=0.01, s=footnote_r, fontsize=4, fontweight='light', horizontalalignment='right')

    return pltfig_to_pil(fig)


def pltfig_to_pil(figure: Figure) -> Image:
    """
    Convert a matplotlib figure (e.g. from plt.cfg()) to a PIL image.
    :param (Figure) figure: a `matplotlib.figure.Figure` instance with the image we want to convert to PIL
    :return: a `PIL.Image` object
    """
    # Create a buffer to read the figure data
    _temp_buffer = io.BytesIO()
    atexit.register(_temp_buffer.close)
    # Save figure in buffer
    figure.savefig(_temp_buffer, format='jpg')
    # Read image from buffer and return
    _temp_buffer.seek(0)
    return Image.open(_temp_buffer)


def squarify_img(img: Union[str, Image.Image], target_shape: Optional[int] = None,
                 bg_color: Union[str, float, int] = 'white') -> Image:
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
