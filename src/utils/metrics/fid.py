import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
from IPython import get_ipython
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.deep_fashion import ICRBCrossPoseDataset
from modules.generators.pgpg import PGPGGenerator
from utils.pytorch import matrix_sqrt, cov


def _frechet_distance(x_mean: Tensor, y_mean: Tensor, x_cov: Tensor, y_cov: Tensor) -> Tensor:
    """
    Method for returning the Fréchet distance between multivariate Gaussians, parameterized by their means and
    covariance matrices.
    :param x_mean: the mean of the first Gaussian, (n_vars)
    :param y_mean: the mean of the second Gaussian, (n_vars)
    :param x_cov: the covariance matrix of the first Gaussian, (n_vars, n_vars)
    :param y_cov: the covariance matrix of the second Gaussian, (n_vars, n_vars)
    :return: a torch.Tensor object containing the Frechet distance of the two multivariate Gaussian distributions
    """
    return torch.norm(x_mean - y_mean) ** 2 + torch.trace(x_cov + y_cov - 2 * matrix_sqrt(x_cov @ y_cov))


class FID(nn.Module):
    """
    FID Class:
    This class is used to compute the Fréchet Inception Distance (FID) between two given image sets.
    """

    InceptionV3Transforms = transforms.Compose([
        # transforms.Resize(299),
        # transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, chkpts_root: str = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints',
                 device: str = 'cpu', n_samples: int = 512, batch_size: int = 8, crop_fc: bool = True):
        """
        FID class constructor.
        :param chkpts_root: absolute path to model checkpoints directory
        :param device: the device type on which to run the Inception model (defaults to 'cpu')
        :param n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                          number gets, the more accurate the metric is)
        :param batch_size: the number of samples to precess at each loop
        :param crop_fc: set to True to crop FC layer from Inception v3 network
        """
        super(FID, self).__init__()
        self.inside_colab = 'google.colab' in sys.modules or \
                            'google.colab' in str(get_ipython()) or \
                            'COLAB_GPU' in os.environ
        if self.inside_colab:
            chkpts_root = '/content/drive/MyDrive/Model Checkpoints'
            device = 'cuda'

        chkpt_path = os.path.join(chkpts_root, 'inception_v3_google-1a9a5a14.pth')

        # Confirm checkpoint exists
        assert os.path.exists(chkpt_path)
        # Instantiate Inception v3 model
        self.inception = inception_v3(pretrained=False, init_weights=False)
        self.inception.load_state_dict(torch.load(chkpt_path))
        self.inception \
            .to(device) \
            .eval()
        # Cutoff FC layer from Inception model since we do not want classification, just the feature embedding
        if crop_fc:
            self.inception.fc = nn.Identity()

        # Save params in instance
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size

    # noinspection DuplicatedCode
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None) -> Tensor:
        """
        Compute the Fréchet Inception Distance between random $self.n_samples$ images from the given dataset and same
        number of images generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :return: a scalar torch.Tensor object containing the computed FID value
        """
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        cur_samples = 0
        real_embeddings_list = []
        fake_embeddings_list = []
        for real_samples in tqdm(dataloader, total=self.n_samples // self.batch_size):
            if cur_samples >= self.n_samples:
                break

            # Compute real embeddings
            target_output = real_samples[target_index] if target_index is not None else real_samples
            target_output = target_output.to(self.device)
            real_embeddings = self.inception(target_output)
            real_embeddings_list.append(real_embeddings.detach().cpu())

            cur_batch_size = len(target_output)

            # Compute fake embeddings
            gen_inputs = [real_samples[_i] for _i in condition_indices] if condition_indices is not None else \
                torch.randn(cur_batch_size, z_dim)
            gen_inputs = [gen_input.to(self.device) for gen_input in gen_inputs] if condition_indices is not None \
                else gen_inputs.to(self.device)
            fake_output = gen(*gen_inputs)
            if type(fake_output) == tuple or type(fake_output) == list:
                fake_output = fake_output[-1]
            fake_embeddings = self.inception(fake_output)
            fake_embeddings_list.append(fake_embeddings.detach().cpu())

            cur_samples += cur_batch_size

        # Compute sample mean and covariance of real embeddings
        real_embeddings = torch.cat(real_embeddings_list, dim=0)
        real_embeddings_mean = torch.mean(real_embeddings, dim=0)
        real_embeddings_cov = cov(real_embeddings)

        # Compute sample mean and covariance of fake embeddings
        fake_embeddings = torch.cat(fake_embeddings_list, dim=0)
        fake_embeddings_mean = torch.mean(fake_embeddings, dim=0)
        fake_embeddings_cov = cov(fake_embeddings)

        return _frechet_distance(real_embeddings_mean, fake_embeddings_mean,
                                 real_embeddings_cov, fake_embeddings_cov)


if __name__ == '__main__':
    _fid = FID(n_samples=2, batch_size=1)
    _dataset = ICRBCrossPoseDataset(image_transforms=FID.InceptionV3Transforms, pose=True)
    _gen = PGPGGenerator(c_in=6, c_out=3)
    fid = _fid(_dataset, _gen, target_index=1, condition_indices=(0, 2))
    print(fid)