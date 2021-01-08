import atexit
import io
import json
import os
from shutil import copy2
from typing import Union, Optional

import matplotlib
import matplotlib.font_manager
from PIL import Image
from matplotlib.figure import Figure

from utils.filesystems.gdrive.remote import GDriveFolder


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
        os.system(f'mkdir -p {sf_fonts_folder}')
        for f in sf.files:
            if not f.name.endswith('ttf'):
                continue
            # Copy file to matplotlib folder
            if not os.path.exists(f'{matplotlib_ttf_path}/{f.name}'):
                new_ttf_files.append(copy2(f.path, matplotlib_ttf_path))
            # Copy to system fonts folder
            if not os.path.exists(f'{sf_fonts_folder}/{f.name}'):
                copy2(f.path, sf_fonts_folder)
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
