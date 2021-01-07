from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from datasets.deep_fashion import ICRBCrossPoseDataset, ICRBDataset
from modules.generators.pgpg import PGPGGenerator
from utils.filesystems.gdrive import ColabCapsule, ColabFolder
from utils.ifaces import FilesystemFolder
from utils.metrics.fid import FID


class ManifoldEstimator:
    """
    ManifoldEstimator Class:
    Estimates the manifold of given feature vectors.
    Source: https://github.com/kynkaat/improved-precision-and-recall-metric
    """

    def __init__(self, embeddings: Tensor, row_batch_size: int, col_batch_size: int, k: int = 3,
                 clamp_to_percentile=None, eps=1e-5):
        """
        Estimate the manifold of given feature vectors.
        :param (Tensor) embeddings: ImageNet embeddings as a torch.Tensor object of shape (n_vectors, n_features)

            Args:
                feature_vectors (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                k (int): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        n_vectors = len(embeddings)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = embeddings

        # Estimate manifold of features by calculating distances to k-NN of each sample
        self.dtype = np.float16
        self.kth_distance = np.zeros(n_vectors, dtype=self.dtype)
        for begin1 in range(0, n_vectors, row_batch_size):
            end1 = min(begin1 + row_batch_size, n_vectors)
            row_batch = embeddings[begin1:end1]

            batch_distances = np.zeros([end1 - begin1, n_vectors], dtype=self.dtype)
            for begin2 in range(0, n_vectors, col_batch_size):
                end2 = min(begin2 + col_batch_size, n_vectors)
                col_batch = embeddings[begin2:end2]

                # Compute distances between batches.
                batch_distances[:, begin2:end2] = \
                    self.__class__._batch_pairwise_distances(row_batch, col_batch).numpy().astype(self.dtype)

            # Find the k-nearest neighbor from the current batch.
            self.kth_distance[begin1:end1] = np.partition(batch_distances, range(0, k + 1), axis=1)[:, k]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.kth_distance, clamp_to_percentile, axis=0)
            self.kth_distance[self.kth_distance > max_distances] = 0

    @staticmethod
    def _batch_pairwise_distances(u: Tensor, v: Tensor) -> Tensor:
        """
        Compute pairwise distances between two batches of feature vectors.
        :param u: 1st input vector of shape (batch_size, n_features)
        :param v: 2nd input vector of shape (batch_size, n_features)
        :return: a torch.Tensor object of shape (batch_size, batch_size)
        """
        # Squared norms of each row in U and V.
        norm_u = torch.norm(u, dim=1) ** 2
        norm_v = torch.norm(v, dim=1) ** 2

        # Pairwise squared Euclidean distances.
        d = torch.abs(norm_u - 2 * (u @ v.transpose(-1, -2)) + norm_v)
        return d

    def evaluate(self, eval_embeddings: Tensor) -> Tensor:
        """
        Evaluate if new feature vectors are at the manifold.
        :param eval_embeddings: the embeddings to test which lie in self manifold. a torch.Tensor object of shape
                                (n_vectors, n_features)
        :return: a torch.Tensor object of shape (n_vectors, 1)
        """
        num_eval_images = eval_embeddings.shape[0]
        num_ref_images = self.kth_distance.shape[0]

        batch_predictions = np.zeros([num_eval_images, ], dtype=np.int32)
        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_embeddings[begin1:end1]

            distance_batch = np.zeros([end1 - begin1, num_ref_images], dtype=self.dtype)
            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[:, begin2:end2] = \
                    self.__class__._batch_pairwise_distances(feature_batch, ref_batch).numpy().astype(self.dtype)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch <= self.kth_distance
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

        return torch.from_numpy(batch_predictions).float()


class F1(FID):
    """
    F1 Class:
    This class is used to compute the Precision, Recall and F1 metrics between real and generated images as proposed in
    the paper "Improved Precision and Recall Metric for Assessing Generative Models".
    """

    def __init__(self, model_fs_folder_or_root: Union[FilesystemFolder, str], device: torch.device or str = 'cpu',
                 n_samples: int = 1024, batch_size: int = 8):
        """
        FID class constructor.
        :param (FilesystemFolder or str) model_fs_folder_or_root: absolute path to model checkpoints directory or
                                                                  FilesystemFolder instance for cloud-synced models
        :param (str) device: the device type on which to run the Inception model (defaults to 'cpu')
        :param (int) n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                          number gets, the more accurate the metric is)
        :param (int) batch_size: the number of samples to precess at each loop
        """
        super(F1, self).__init__(model_fs_folder_or_root=model_fs_folder_or_root, device=device,
                                 n_samples=n_samples, batch_size=batch_size)

    # noinspection PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, gen_transforms: transforms.Compose,
                target_index: Optional[int] = None, condition_indices: Optional[Union[int, tuple]] = None,
                z_dim: Optional[int] = None, k: int = 3, row_batch_size: int = 8, col_batch_size: int = 8,
                show_progress: bool = True, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the F1 score between random $self.n_samples$ images from the given dataset and same number of images
        generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images. Attention: no transforms should be
                        applied when __getitem__ is called since the transforms are different on Inception v3
        :param gen: the Generator network
        :param gen_transforms: the torchvision transforms on which the generator was trained
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param k: the "k" of k-NN; here used as the radius of the hypersphere of points
        :param row_batch_size: see https://github.com/kynkaat/improved-precision-and-recall-metric
        :param col_batch_size: see https://github.com/kynkaat/improved-precision-and-recall-metric
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :return: a tuple containing (f1, precision, recall) as torch.Tensor objects
        """
        # Extract ImageNET embeddings
        real_embeddings, fake_embeddings = self.get_embeddings(dataset, gen=gen, gen_transforms=gen_transforms,
                                                               target_index=target_index, z_dim=z_dim,
                                                               condition_indices=condition_indices,
                                                               show_progress=show_progress)
        # Initialize manifolds
        real_manifold = ManifoldEstimator(real_embeddings, row_batch_size, col_batch_size, k=k)
        fake_manifold = ManifoldEstimator(fake_embeddings, row_batch_size, col_batch_size, k=k)
        # Compute precision (i.e. how many points from fake_embeddings fall into the real_embeddings manifold)
        precision = real_manifold.evaluate(fake_embeddings).mean()
        # Compute recall (i.e. how many points from real_embeddings fall into the fake_embeddings manifold)
        recall = fake_manifold.evaluate(real_embeddings).mean()
        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1, precision, recall


if __name__ == '__main__':
    # Init Google Drive staff
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _capsule = ColabCapsule(local_gdrive_root=_local_gdrive_root)
    _groot = ColabFolder.root(capsule_or_fs=_capsule).subfolder_by_name('Models')

    f1_calculator = F1(model_fs_folder_or_root=_groot, n_samples=5, batch_size=1)
    _dataset = ICRBCrossPoseDataset(image_transforms=None, pose=True)
    _gen = PGPGGenerator(c_in=6, c_out=3, w_in=128, h_in=128)
    _gen_transforms = ICRBDataset.get_image_transforms(target_shape=128, target_channels=3)
    _f1, _precision, _recall = f1_calculator(_dataset, gen=_gen, gen_transforms=_gen_transforms, target_index=1,
                                             condition_indices=(0, 2), k=1, show_progress=True)
    print(_f1, _precision, _recall)
