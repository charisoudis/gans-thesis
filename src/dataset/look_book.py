import json
import os
import re
from shutil import copyfile
from time import sleep

import click
from tqdm import tqdm

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
            self.logger.info('[initial_scraping] STARTING')
            self.initial_scraping()
            self.logger.info('[initial_scraping] DONE')
            self.needs_initial_scraping = not os.path.exists(self.img_dir_path)

        self.items_count = 0 if self.needs_initial_scraping else len(next(os.walk(self.img_dir_path))[1])

        self.needs_initial_scraping_deep_fashion = self.items_count <= 8726
        if self.needs_initial_scraping_deep_fashion and \
                click.confirm('Do you want to perform the initial scraping of DeepFashion to expand LookBook with ' +
                              'more pairs now?', default=True):
            sleep(0.5)
            self.logger.info('[initial_scraping_deep_fashion] STARTING')
            sleep(0.5)
            self.initial_scraping_deep_fashion()
            self.logger.info('[initial_scraping_deep_fashion] DONE')
            self.needs_initial_scraping_deep_fashion = self.items_count <= 8726

    def initial_scraping(self, squarify: bool = True) -> None:
        """
        Transfers all images from initial image directory (as provided by the authors) into a directory structure where
        all the images of the same product id (pid) are placed into the same directory.
        :param squarify: set to True to have the original images expanded to square images and then resized to 256x256px
        """
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

    def initial_scraping_deep_fashion(self) -> None:
        """
        Method to scrape DeepFashion ICRB dataset for items that contain image groups with flat.jpg images. These image
        groups can be copied over to LookBook Img root as new items since the can be used in a similar manner as the
        LookBook dataset's images. Transferred images are renamed to match naming style of LookBook item images.
        """
        # Define DeepFashion's Img root
        df_img_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark/Img'
        if not click.confirm(f'DeepFashion img root: {df_img_root}. Correct?', default=True) or \
                not os.path.exists(df_img_root):
            df_img_root = input('DeepFashion img root: ')
            assert os.path.exists(df_img_root), f'Provided df_img_root={df_img_root} NOT FOUND'
        items_info_path = f'{df_img_root}/items_info.json'
        assert os.path.exists(items_info_path), f'items_info.json NOT FOUND at DeepFashion Img root. Dataset should' + \
                                                f'have been scraped before. (tried path: {items_info_path})'

        # Load items info and look for image groups
        with open(items_info_path) as json_fp:
            items_info = json.load(json_fp)
        image_groups = items_info['image_groups']

        self.logger.debug(f'[initial_scraping_deep_fashion] Found {len(image_groups)} in total')

        # Get all useful image groups
        dt_groups = []
        dt_groups_count = 0
        for path, images in image_groups.items():
            flat_images = [_i for _i in images if _i.endswith('flat.jpg')]
            if 0 == len(flat_images):
                continue
            assert len(flat_images) == 1, f'len(flat_images)={len(flat_images)} > 1!'

            dt_groups.append({
                'src_path': f'{df_img_root}/{path}',
                'src_images': images,
            })
            dt_groups_count += 1

        self.logger.debug(f'[initial_scraping_deep_fashion] Found {dt_groups_count} out of {len(image_groups)}' +
                          f' items with a flat.jpg image.')
        with tqdm(total=dt_groups_count, colour='yellow') as progress_bar:
            # Copy images in LookBook under Img root
            pid = self.items_count
            for i, dt_group in enumerate(dt_groups):
                src_dir_path = dt_group['src_path']
                dst_dir_path = f'{self.img_dir_path}/id_{str((pid + i)).zfill(8)}'

                src_images = sorted(dt_group['src_images'], key=lambda _i: int(_i.split(sep='_', maxsplit=1)[0]))
                src_human_images = [_i for _i in src_images if _i.endswith('jpg') and not _i.endswith('flat.jpg')]
                src_flat_image = [_i for _i in src_images if _i.endswith('flat.jpg')][0]

                dst_human_images = [f'{dst_dir_path}/{_i}.jpg' for _i, _ in enumerate(src_human_images)]
                dst_flat_image = f'{dst_dir_path}/flat.jpg'

                src_human_images = [f'{src_dir_path}/{_i}' for _i in src_human_images]
                src_human_images = ['_'.join(_i.rsplit('/', 1)) for _i in src_human_images]
                src_flat_image = f'{src_dir_path}/{src_flat_image}'
                src_flat_image = '_'.join(src_flat_image.rsplit('/', 1))

                # Create destination folder and copy files
                if not os.path.exists(dst_dir_path):
                    os.mkdir(dst_dir_path)
                for _src, _dst in zip(src_human_images, dst_human_images):
                    copyfile(_src, _dst)
                copyfile(src_flat_image, dst_flat_image)

                # Stamp destination folder with DeepFashion data
                with open(f'{dst_dir_path}/.src', 'w') as fp:
                    fp.write(f'{src_flat_image.replace("/" + os.path.basename(src_flat_image), "")}\n')

                self.items_count += 1
                # self.logger.debug(f'[initial_scraping_deep_fashion] {dst_dir_path}: DONE')
                progress_bar.update()

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping LookBook images:
        Visits every item directory, process its images and saves image / pairs information a JSON file named
        `item_dt_info.json`.
        """
        id_dirs = next(os.walk(self.img_dir_path))[1]
        with tqdm(total=len(id_dirs), colour='yellow') as progress_bar:
            for id_dir in id_dirs:
                id_dir_path = f'{self.img_dir_path}/{id_dir}'
                images = os.listdir(id_dir_path)
                assert 'flat.jpg' in images
                images = sorted([_i for _i in images if _i.endswith('.jpg') and _i != 'flat.jpg'],
                                key=lambda _i: int(_i.replace('.jpg', '')))
                dt_image_pairs = [(_i, 'flat.jpg') for _i in images]
                dt_info = {
                    'id': int(id_dir.replace('id_', '')),
                    'path': f'/{id_dir}',
                    'flat_images': [
                        'flat.jpg'
                    ],
                    'flat_images_count': 1,
                    'human_images': images,
                    'human_images_count': len(images),
                    'dt_image_pairs': dt_image_pairs,
                    'dt_image_pairs_count': len(dt_image_pairs),
                }

                with open(f'{id_dir_path}/item_dt_info.json', 'w') as json_fp:
                    json.dump(dt_info, json_fp, indent=4)
                # self.logger.debug(f'{id_dir_path}: [DONE]')
                progress_bar.update()

    def backward(self) -> None:
        """
        Method for completing a backward pass in scraping LookBook images:
        Similar to DeepFashion scraper's backward pass, recursively visits all directories under image root merging
        information saved in JSON files found inside children directories.
        """
        # Initialize aggregator
        id_dirs = next(os.walk(self.img_dir_path))[1]
        items_dt_info = {
            'id': 'Img',
            'path': '',
            'flat_images': [],
            'flat_images_count': 0,
            'human_images': [],
            'human_images_count': 0,
            'dt_image_pairs': [],
            'dt_image_pairs_count': 0,
        }
        # Start merging
        with tqdm(total=len(id_dirs), colour='yellow') as progress_bar:
            for id_dir in id_dirs:
                id_dir_path = f'{self.img_dir_path}/{id_dir}'
                item_dt_info_path = f'{id_dir_path}/item_dt_info.json'
                assert os.path.exists(item_dt_info_path), f'item_dt_info_path={item_dt_info_path}: NOT FOUND'
                with open(item_dt_info_path) as json_fp:
                    item_dt_info = json.load(json_fp)

                # Prefix images
                file_prefix = item_dt_info['path'].lstrip('/')
                for _i, _name in enumerate(item_dt_info['flat_images']):
                    item_dt_info['flat_images'][_i] = f'{file_prefix}/{_name}'
                for _i, _name in enumerate(item_dt_info['human_images']):
                    item_dt_info['human_images'][_i] = f'{file_prefix}/{_name}'
                for _i, _pair in enumerate(item_dt_info['dt_image_pairs']):
                    item_dt_info['dt_image_pairs'][_i][0] = f'{file_prefix}/{_pair[0]}'
                    item_dt_info['dt_image_pairs'][_i][1] = f'{file_prefix}/{_pair[1]}'

                # Merge item in aggregated items info
                items_dt_info['flat_images'] += item_dt_info['flat_images']
                items_dt_info['flat_images_count'] += item_dt_info['flat_images_count']
                items_dt_info['human_images'] += item_dt_info['human_images']
                items_dt_info['human_images_count'] += item_dt_info['human_images_count']
                items_dt_info['dt_image_pairs'] += item_dt_info['dt_image_pairs']
                items_dt_info['dt_image_pairs_count'] += item_dt_info['dt_image_pairs_count']

                progress_bar.update()

        with open(f'{self.img_dir_path}/items_dt_info.json', 'w') as json_fp:
            json.dump(items_dt_info, json_fp, indent=4)

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True) -> None:
        """
        Entry point of class.
        :param forward_pass: set to True to run scraper's forward pass (create item_dt_info.json files in item dirs)
        :param backward_pass: set to True to run scraper's backward pass (recursively merge items JSON files)
                              Note: if $forward_pass$ is set to True, then $backward_pass$ is also set to True.
        """
        scraper = LookBookScraper()
        scraper.logger.info(f'SCRAPE DIR = {scraper.img_dir_path}')
        if forward_pass:
            # Forward pass
            scraper.logger.info('[forward] STARTING')
            scraper.forward()
            scraper.logger.info('[forward] DONE')
            backward_pass = True
        # Backward pass
        if backward_pass:
            scraper.logger.info('[backward] STARTING')
            scraper.backward()
            scraper.logger.info('[backward] DONE')
        scraper.logger.info('DONE')


if __name__ == '__main__':
    if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
        LookBookScraper.run(forward_pass=True, backward_pass=True)
