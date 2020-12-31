import os
import re
from shutil import copyfile

import click

from utils.command_line_logger import CommandLineLogger
from utils.data import squarify_img


class LookBookScraper:
    """
    LookBookScraper Class:
    This class is used to scrape LookBook dataset's images.
    """

    def __init__(self, root: str = '/data/Datasets/LookBook'):
        """
        LookBookScraper class constructor.
        :param root: LookBook dataset's root directory path
        """
        self.logger = CommandLineLogger(log_level='debug')
        self.initial_img_dir_path = f'{root}/ImgHQ'
        self.img_dir_path = f'{root}/Img'
        self.needs_initial_scraping = not os.path.exists(self.img_dir_path)

        if self.needs_initial_scraping and \
                click.confirm('Do you want to perform the initial scraping of the dataset now?', default=True):
            self.initial_scraping()

    def initial_scraping(self, squarify: bool = True) -> None:
        """
        Transfers all images from initial image directory (as provided by the authors) into a directory structure where
        all the images of the same product id (pid) are placed into the same directory.
        :param squarify: set to True to have the original images expanded to square images and then resized to 256x256px
        """
        self.logger.info('initial_scraping(): [STARTING]')
        # Create output image directory
        if os.path.exists(self.img_dir_path):
            os.rmdir(self.img_dir_path)
        os.mkdir(self.img_dir_path)
        # Define regex
        img_regex = re.compile('PID(\d+)_CLEAN(\d)_IID(\d+).jpg')
        # Fetch image names as list
        initial_image_list = os.listdir(self.initial_img_dir_path)
        initial_image_list = sorted(initial_image_list)
        current_pid = None
        counter = None
        pid_dir_path = None
        for img in initial_image_list:
            # Get image data from filename
            pid, clean, iid = img_regex.findall('%s' % img)[0]
            pid = int(pid)
            # iid = int(iid)
            is_flat = '1' == clean
            # Initiate new Img/id_<8-digit_product_id> directory
            if pid != current_pid:
                # Before creating the new dir, check the previous contains flat.jpg and at least one other image
                if pid_dir_path is not None:
                    _, _, _files = next(os.walk(pid_dir_path))
                    assert len(_files) >= 2
                    assert 'flat.jpg' in _files
                # Create new directory
                pid_dir_path = f'{self.img_dir_path}/id_{str(pid).zfill(8)}'
                if not os.path.exists(pid_dir_path):
                    os.mkdir(pid_dir_path)
                self.logger.debug(f'initial_scraping(): New dir created at {pid_dir_path}')
                # Reset counter
                counter = 0
                current_pid = pid
            # Copy files
            src_img_filepath = f'{self.initial_img_dir_path}/{img}'
            dst_img_filename = str(counter) if not is_flat else 'flat'
            dst_img_filepath = f'{self.img_dir_path}/{os.path.basename(pid_dir_path)}/{dst_img_filename}.jpg'
            if squarify:
                result = squarify_img(src_img_filepath, target_shape=256, bg_color='white')
                result.save(dst_img_filepath, quality=95)
            else:
                copyfile(f'{src_img_filepath}', dst_img_filepath)
            # self.logger.debug(f'{src_img_filepath} --> {dst_img_filepath}')
            counter += 1

        self.logger.info('initial_scraping(): [DONE]')

    def forward(self):
        # TODO
        pass

    def backward(self):
        # TODO
        pass


if __name__ == '__main__':
    scraper = LookBookScraper()
