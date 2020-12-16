import os
import pathlib
import uuid
from typing import Union, List, Tuple

import requests
from torch import Tensor

from utils.tensor import save_tensor_to_image_file


class DensePoseDetector:
    """
    DensePoseDetector Class:
    This class is used to create pose maps for input images using DensePose library.
    """

    def __init__(self, dense_pose_api_root: str = 'http://localhost:5000/'):
        """
        DensePoseDetector class constructor.
        """
        self.dense_pose_api_root = dense_pose_api_root
        self.current_dir_root = str(pathlib.Path(__file__).parent.absolute())

    def make_request_for_single_file(self, filepath_or_tensor: Union[str, Tensor]) -> Tuple[requests.Response, str]:
        """
        Makes the HTTP POST request to running DensePose API (inside Docker container) for 1 image file or torch.Tensor.
        :param filepath_or_tensor: an image file path or torch.Tensor object (which is first saved as JPG image locally)
        :return: a tuple containing the requests.Response object with API's response (returned image can be accessed via
                 response.content) and the filepath of the image file used to make the request
        """
        # Check if is file path
        if type(filepath_or_tensor) == Tensor:
            random_filename = str(uuid.uuid4()) + '.jpg'
            random_filename = f'{self.current_dir_root}/{random_filename}'
            save_tensor_to_image_file(filepath_or_tensor, random_filename)
            filepath = random_filename
        else:
            filepath = filepath_or_tensor
        # Send request
        request_file = dict(img_file=open(filepath))
        response = requests.post(self.dense_pose_api_root, files=request_file)
        # Delete temp file (if created one)
        if type(filepath_or_tensor) == Tensor:
            os.remove(filepath)
        return response, filepath

    def make_request(self, *filepaths_or_tensors: Union[str, Tensor], save_responses: bool = False) \
            -> List[Union[str, bytes]]:
        """
        Makes the HTTP POST request to running DensePose API (inside Docker container).
        :param filepaths_or_tensors: a list of image file paths or torch.Tensor objects (which are firstly saved as
                                     images locally)
        :param save_responses: set to True to save responses contents to same filepaths with '_IUV' appended to
                               respective filenames and return the list of created files paths. set to False to have
                               responses contents returned as list of strings.
        :return: a requests.Response object containing API's response
        """
        # Process files sequentially
        _result_list = []
        for filepaths_or_tensor in filepaths_or_tensors:
            # Make the request
            response, filepath = self.make_request_for_single_file(filepaths_or_tensor)
            # Save the received file
            if save_responses:
                new_filepath = filepath.replace('.jpg', '_IUV.png')
                with open(new_filepath, 'wb') as fp:
                    fp.write(response.content)
                # Append to saved files
                _result_list.append(new_filepath)
            else:
                # Append to saved bytes arrays
                _result_list.append(response.content)

        return _result_list

    @staticmethod
    def from_files(*filepaths: str) -> None:
        dpd = DensePoseDetector()
        new_filepaths = dpd.make_request(*filepaths, save_responses=True)
        print(new_filepaths)

    @staticmethod
    def from_tensors(*tensors: Tensor):
        dpd = DensePoseDetector()
        new_filepaths = dpd.make_request(*tensors, save_responses=True)
        print(new_filepaths)


if __name__ == '__main__':
    dataset_root = '/data/Datasets/DeepFashion/In-shop Clothes Retrieval Benchmark/Anno/dense pose/img'
    img_filepath = f'{dataset_root}/MEN/Denim/id_00000182/01_1_front.jpg'
    DensePoseDetector.from_files(img_filepath)
