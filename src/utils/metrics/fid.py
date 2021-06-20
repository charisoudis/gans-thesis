import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from datasets.deep_fashion import ICRBCrossPoseDataset, ICRBDataset
from modules.generators.pgpg import PGPGGenerator
from modules.inception import InceptionV3
from utils.dep_free import get_tqdm
from utils.filesystems.local import LocalCapsule, LocalFolder
from utils.ifaces import FilesystemFolder, Freezable
from utils.pytorch import matrix_sqrt, cov, ToTensorOrPass


def _frechet_distance(x_mean: Tensor, y_mean: Tensor, x_cov: Tensor, y_cov: Tensor) -> Tensor:
    """
    Method for returning the Fréchet distance between multivariate Gaussians, parameterized by their means and
    covariance matrices.
    :param (Tensor) x_mean: the mean of the first Gaussian, (n_vars)
    :param (Tensor) y_mean: the mean of the second Gaussian, (n_vars)
    :param (Tensor) x_cov: the covariance matrix of the first Gaussian, (n_vars, n_vars)
    :param (Tensor) y_cov: the covariance matrix of the second Gaussian, (n_vars, n_vars)
    :return: a `torch.Tensor` object containing the Frechet distance of the two multivariate Gaussian distributions
    """
    return torch.norm(x_mean - y_mean) ** 2 + torch.trace(x_cov + y_cov - 2 * matrix_sqrt(x_cov @ y_cov))


class FID(nn.Module):
    """
    FID Class:
    This class is used to compute the Fréchet Inception Distance (FID) between two given image sets.
    """

    # These are the Inception v3 image transforms
    InceptionV3Transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        ToTensorOrPass(renormalize=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Keep the Inception network in a static variable to avoid re-initializing it on sub-classes
    InceptionV3Cropped = None  # for feature embedding (e.g. for FID, Precision, Recall, F1)
    InceptionV3Classifier = None  # for classification (e.g. for IS)

    # Keep embeddings in memory
    LastRealEmbeddings = None
    LastFakeEmbeddings = None

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, device: torch.device or str = 'cpu',
                 n_samples: int = 512, batch_size: int = 8):
        """
        FID class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` instance for cloud or
                                                           locally saved models using the same API
        :param (str) device: the device type on which to run the Inception model (defaults to 'cpu')
        :param (int) n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                                number gets, the more accurate the metric is)
        :param (int) batch_size: the number of samples to precess at each loop
        """
        super(FID, self).__init__()
        self.tqdm = get_tqdm()

        # Instantiate Inception v3 model(s)
        if FID.InceptionV3Cropped is None:
            FID.InceptionV3Cropped = InceptionV3(model_fs_folder_or_root, chkpt_step='1a9a5a14',
                                                 crop_fc=False).inception_v3
            # Deep copy last FC before cropping network
            inception_fc = nn.Linear(in_features=FID.InceptionV3Cropped.fc.in_features,
                                     out_features=FID.InceptionV3Cropped.fc.out_features, bias=True)
            inception_fc.load_state_dict(FID.InceptionV3Cropped.fc.state_dict())
            FID.InceptionV3Cropped.fc = nn.Identity()  # crop network by removing FC
            FID.InceptionV3Cropped = FID.InceptionV3Cropped.to(device).eval()
            # Define the Inception v3 classifier
            FID.InceptionV3Classifier = nn.Sequential(
                FID.InceptionV3Cropped,
                inception_fc,
                nn.Softmax(dim=1)
            ).to(device).eval()

        # Save params in instance
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size

    # noinspection DuplicatedCode
    def get_embeddings(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                       condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                       show_progress: bool = True, desc: str = "FID") -> Tuple[Tensor, Tensor]:
        """
        Computes ImageNet embeddings of a batch of real and fake images based on Inception v3 classifier.
        :param (Dataset) dataset: the torch.utils.data.Dataset instance to access dataset of real images
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :param (str) desc: tqdm `desc` parameter (is printed on the left of percentage indicator)
        :return: a tuple containing one torch.Tensor object of shape (batch_size, n_features) for each of real, fake
                 images
        """
        # Freeze Generator
        assert isinstance(gen, Freezable), 'Generator should implement utils.ifaces.Freezable'
        with gen.frozen():

            # Create the dataloader instance
            dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
            if self.device == 'cuda:0' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Extract embeddings
            cur_samples = 0
            real_embeddings_list = []
            fake_embeddings_list = []
            break_after = False
            for real_samples in self.tqdm(dataloader, total=int(math.ceil(self.n_samples / self.batch_size)),
                                          disable=not show_progress, desc=desc):
                if cur_samples >= self.n_samples:
                    break_after = True

                if hasattr(gen, 'resolution'):
                    real_samples = transforms.Resize(size=gen.resolution)(real_samples)

                # Compute real embeddings
                target_output = real_samples[target_index] if target_index else real_samples
                target_output = target_output.to(self.device)
                real_embeddings = FID.InceptionV3Cropped(FID.InceptionV3Transforms(target_output))
                real_embeddings_list.append(real_embeddings.detach().cpu())

                cur_batch_size = len(target_output)

                # Compute fake embeddings
                gen_inputs = [real_samples[_i].to(self.device) for _i in condition_indices] \
                    if condition_indices else [torch.randn(cur_batch_size, z_dim, device=self.device), ]
                # gen_inputs = [gen_transforms(gen_input).to(self.device) for gen_input in gen_inputs] \
                #     if condition_indices is not None else gen_inputs.to(self.device)
                fake_output = gen(*gen_inputs)
                fake_output_type = type(fake_output)
                if fake_output_type != torch.Tensor and (type(fake_output) == tuple or type(fake_output) == list):
                    fake_output = fake_output[-1]
                # ATTENTION: In order to pass generator's output through Inception we must re-normalize tensor stats!
                # Generator output images in the range [-1, 1], since it uses a Tanh() activation layer, whereas
                # Inception v3 receives tensors with its custom normalization. Solutions: 1) Invert normalization in
                # gen_transforms and then pass the image through the Inception transforms | 2) Use the new
                # ToTensorOrPass() transform fake_output = gen_transforms_inv(fake_output)
                fake_embeddings = FID.InceptionV3Cropped(FID.InceptionV3Transforms(fake_output))
                fake_embeddings_list.append(fake_embeddings.detach().cpu())

                cur_samples += cur_batch_size
                if break_after:
                    break

        return torch.cat(real_embeddings_list, dim=0), torch.cat(fake_embeddings_list, dim=0)

    # noinspection PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                show_progress: bool = True, **kwargs) -> Tensor:
        """
        Compute the Fréchet Inception Distance between random $self.n_samples$ images from the given dataset and same
        number of images generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images. Attention: no transforms should be
                        applied when __getitem__ is called since the transforms are different on Inception v3
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :return: a scalar torch.Tensor object containing the computed FID value
        """
        # Extract ImageNET embeddings
        real_embeddings, fake_embeddings = self.get_embeddings(dataset, gen=gen, target_index=target_index, z_dim=z_dim,
                                                               condition_indices=condition_indices,
                                                               show_progress=show_progress)
        FID.LastRealEmbeddings = real_embeddings.clone()
        FID.LastFakeEmbeddings = fake_embeddings.clone()
        # Compute sample means and covariance matrices
        real_embeddings_mean = torch.mean(real_embeddings, dim=0)
        fake_embeddings_mean = torch.mean(fake_embeddings, dim=0)
        real_embeddings_cov = cov(real_embeddings)
        fake_embeddings_cov = cov(fake_embeddings)
        # Compute Frechet distance of embedding vectors and return
        return _frechet_distance(real_embeddings_mean, fake_embeddings_mean, real_embeddings_cov, fake_embeddings_cov)


# noinspection DuplicatedCode
if __name__ == '__main__':
    # Init Google Drive stuff
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _groot = LocalFolder.root(LocalCapsule(_local_gdrive_root))
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Setup evaluation dataset
    _target_shape = 128
    _target_channels = 3
    _dataset = ICRBCrossPoseDataset(dataset_fs_folder_or_root=_datasets_groot,
                                    image_transforms=ICRBDataset.get_image_transforms(_target_shape, _target_channels))

    # Initialize Generator
    _gen = PGPGGenerator(c_in=2 * _target_channels, c_out=_target_channels, w_in=_target_shape, h_in=_target_shape)

    # Evaluate Generator using FID
    _fid = FID(model_fs_folder_or_root=_models_groot, n_samples=2, batch_size=1)
    fid = _fid(_dataset, _gen, target_index=1, condition_indices=(0, 2), show_progress=True)
    print(fid)
