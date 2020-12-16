import glob
import os
from typing import Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageToImageDataset(Dataset):

    def __init__(self, root_dir: str, image_transforms: Optional[transforms.Compose] = None, mode: str = 'train'):
        """
        ImageToImageDataset class constructor.
        :param root_dir: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param mode: string denoting the running mode (supported: 'train', 'eval')
        """
        super(ImageToImageDataset, self).__init__()

        # Create file lists
        self.files_a = sorted(glob.glob(os.path.join(root_dir, '%sA' % mode) + '/*.*'))
        self.files_a_len = len(self.files_a)
        self.files_b = sorted(glob.glob(os.path.join(root_dir, '%sA' % mode) + '/*.*'))
        self.files_b_len = len(self.files_b)
        if self.files_a_len > self.files_b_len:
            # We want len(files_B) >= len(files_A)
            self.files_a, self.files_b = self.files_b, self.files_a
            self.files_a_len, self.files_b_len = self.files_b_len, self.files_a_len
        # Initialize random permutation
        self.random_perm = self.random_permute()
        # Save transforms
        self.transforms = image_transforms

    def random_permute(self) -> Tensor:
        """
        Get a randomly-permuted set of indices for random selection of files from pile B.
        :return: torch.Tensor with randomly-permuted indices in the range [0, len(files_B]. The number of indices
                 returned is len(files_A) since we supposed that len(files_B) >= len(files_A).
        """
        return torch.randperm(self.files_b_len)[:self.files_a_len]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param index: integer with the current image index that we want to read from disk
        :return: a tuple containing the images from domain A and B, each as a torch.Tensor object
        """
        image_a = Image.open(self.files_a[index % self.files_a_len])
        image_b = Image.open(self.files_b[self.random_perm[index]])
        image_a = self.transforms(image_a)
        image_b = self.transforms(image_b)
        if image_a.shape[0] != 3:
            image_a = image_a.repeat(3, 1, 1)
        if image_b.shape[0] != 3:
            image_b = image_b.repeat(3, 1, 1)
        if index == len(self) - 1:
            # Reshuffle image indices for images from pile B
            self.random_permute()
        return image_a, image_b

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.files_a_len


class ImageToImageDataloader(DataLoader):

    def __init__(self, root_dir: str, image_transforms: Optional[transforms.Compose] = None, mode: str = 'train',
                 batch_size: int = 8, *args):
        """
        ImageToImageDataloader class constructor.
        :param root_dir: the root directory where all image files exist
        :param image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param mode: string denoting the running mode (supported: 'train', 'eval')
        :param batch_size: the number of images batch
        :param args: argument list for torch.utils.data.Dataloader constructor
        """
        super(ImageToImageDataloader, self).__init__(dataset=ImageToImageDataset(root_dir, image_transforms, mode),
                                                     batch_size=batch_size, shuffle=True, *args)
