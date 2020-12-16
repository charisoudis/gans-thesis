import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.utils import make_grid, save_image


def array_to_image(array: np.ndarray) -> Image:
    """
    Feed in numpy array in the range of -1 to 1 and return PIL image
    :param array: input numpy array
    :return: PIL Image object
    """
    return Image.fromarray(((array * 127) + 127).astype('uint8'))


def tensor_to_image(tensor: Tensor) -> Image:
    """
    Convert a flipped channel tensor to a PIL image
    :param tensor: input torch.Tensor
    :return: PIL Image object
    """
    arr = np.transpose(np.array(tensor.detach()), (1, 2, 0))
    arr[arr > 1] = 1
    arr[arr < -1] = -1
    arr = ((arr + 1) * 127.5).astype('uint8')
    return Image.fromarray(arr)


def array_to_tensor(array: np.ndarray, tensor_type: str = 'float32') -> Tensor:
    """
    Convert numpy array to tensor after transposing and float 32 conversion
    :param array: input numpy array
    :param tensor_type: type as string argument to np.astype() method
    :return: torch.Tensor object
    """
    return torch.from_numpy(np.transpose(array.astype(tensor_type), (2, 0, 1)))


def save_tensor_to_image_file(tensor: Tensor, filename: str = 'tensor_to_image.jpg') -> None:
    """
    Saves given tensor to a file named after $filename.
    :param tensor: input tensor as 3D or 4D torch.Tensor object
    :param filename: output file name
    """
    if len(tuple(tensor.shape)) > 3:
        tensor = tensor[0]

    save_image(tensor, filename)


def show_tensor_images(x_real, x_fake):
    """ For visualizing images """
    image_tensor = torch.cat((x_fake[:1, ...], x_real[:1, ...]), dim=0)
    image_tensor = (image_tensor + 1) / 2
    assert isinstance(image_tensor, Tensor)
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat, nrow=1)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_tensor_images_2(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images, an
    size per image, plots and prints the images in an uniform grid.
    """
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
