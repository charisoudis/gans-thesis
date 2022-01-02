import os
import re
from collections import namedtuple, OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from torchvision.models import vgg16
from torchvision.transforms import transforms

from modules.ifaces import IGModule
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.ifaces import FilesystemFolder
from utils.pytorch import ToTensorOrPass


class VGG16(nn.Module, IGModule):
    """
    VGG16 Class:
    This class is used to access and use VGG16 nn.Module but with the additional functionality provided from
    inheriting `utils.gdrive.GDriveModel`. Inheriting GDriveModel we can easily download / upload model checkpoints to
    Google Drive using GoogleDrive API's python client.
    """

    DefaultConfiguration = {'crop_fc': False}

    # These are the VGG16 image transforms
    VGG16Transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToTensorOrPass(renormalize=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, model_gfolder_or_groot: FilesystemFolder, chkpt_step: Optional[int or str] = None,
                 crop_fc: bool = False):
        """
        VGG16 class constructor.
        :param (FilesystemFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                     checkpoints and metrics from/to Google Drive
        :param (str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will be
                                         loaded via `nn.Module().load_state_dict()`
        :param (bool) crop_fc: set to True to crop FC layer from VGG16 network (e.g. to get image embeddings)
        """
        # Initialize interface
        self.DefaultConfiguration['crop_fc'] = crop_fc
        IGModule.__init__(self, model_fs_folder_or_root=model_gfolder_or_groot,
                          log_level=os.getenv('TRAIN_LOG_LEVEL', 'info'))

        # Instantiate InceptionV3 model
        nn.Module.__init__(self)
        self.vgg16 = vgg16(pretrained=False, init_weights=False)

        # Load checkpoint from Google Drive
        if chkpt_step:
            chkpt_filepath = self.fetch_checkpoint(epoch_or_id=chkpt_step, step=None)
            self.logger.info(f'[VGG16] Loading {chkpt_filepath}')
            self.load_state_dict(torch.load(chkpt_filepath, map_location='cpu'))

        # Cutoff FC layer from Inception model when we do not want classification, but feature embedding
        if crop_fc:
            self.vgg16.classifier = nn.Identity()
        self.crop_fc = crop_fc

    def state_dict(self, *args, **kwargs) -> Dict[str, Tensor]:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        return self.vgg16.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        # noinspection PyTypeChecker
        return self.vgg16.load_state_dict(state_dict=state_dict, strict=strict)

    def load_configuration(self, configuration: dict) -> None:
        self.crop_fc = configuration['crop_fc']
        if self.crop_fc:
            self.vgg16.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        This method implements the forward pass through Inception v3 network.
        :param x: the input batch of images as a `torch.Tensor` object
        :return: a `torch.Tensor` object with batch of classifications
        """
        # Update gdrive model state
        self.gforward(x.shape[0])
        # Perform the actual forward pass
        return self.vgg16(x)

    def visualize(self, reproducible: bool = False) -> Image:
        raise NotImplementedError

    def visualize_indices(self, indices: int or Sequence) -> Image:
        raise NotImplementedError


class VGG16Sliced(VGG16):
    """
    VGG16Sliced Class:
    This class is used to access and use VGG16 nn.Module but with the additional flexibility of yielding intermediate
    feature embeddings as in richzhang/PerceptualSimilarity.
    """

    DefaultConfiguration = {'crop_fc': True}

    def __init__(self, model_gfolder_or_groot: FilesystemFolder, chkpt_step: Optional[int or str] = None,
                 requires_grad: bool = False):
        """
        VGG16 class constructor.
        :param (FilesystemFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                     checkpoints and metrics from/to Google Drive
        :param (str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will be
                                         loaded via `nn.Module().load_state_dict()`
        :param (bool) requires_grad: set to True to enable gradients computation
        """
        # Initialize module
        super(VGG16Sliced, self).__init__(model_gfolder_or_groot=model_gfolder_or_groot, chkpt_step=chkpt_step,
                                          crop_fc=True)
        # Slice module
        # Source: https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
        vgg_pretrained_features = self.vgg16.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.vgg16.parameters():
                param.requires_grad = False
            self.requires_grad_(False)
        self.n_channels_list = [64, 128, 256, 512, 512]

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param x: the input batch of images as a `torch.Tensor` object
        :return: a tuple containing the intermediate feature activations as `torch.Tensor` objects
        """
        # Transform image first
        x = VGG16.VGG16Transforms(x)
        # Perform the forward pass
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

    def visualize(self, reproducible: bool = False) -> Image:
        super().visualize(reproducible=reproducible)

    def visualize_indices(self, indices: int or Sequence) -> Image:
        super().visualize_indices(indices=indices)


class LPIPSLoss(nn.Module):
    """
    LPIPSLoss Class:
    This module based on VGG16 will be used for extracting LPIPS metric using VGG-16 and Zhang weighting.
    Takes reference images and corrupted images (STACKED TOGETHER) as an input and outputs the perceptual distance
    between the image pairs.
    Source for LPIPS Metric: https://arxiv.org/abs/1801.03924
    Source 1: https://gist.github.com/shawwn/971d9813a5e45255f507cdfe6981c469
    Source 2: https://github.com/S-aiueo32/lpips-pytorch
    """

    def __init__(self, model_gfolder_or_groot: FilesystemFolder, chkpt_step: Optional[int or str] = None,
                 use_dropout: bool = False, requires_grad: bool = False):
        """
        LPIPSLoss class constructor.
        :param (FilesystemFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                     checkpoints and metrics from/to Google Drive
        :param (str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will be
                                         loaded via `nn.Module().load_state_dict()`
        :param (bool) use_dropout: set to True to use Dropout in Linear layers
        """
        # Initialize module
        super(LPIPSLoss, self).__init__()
        self.vgg16_sliced = VGG16Sliced(model_gfolder_or_groot=model_gfolder_or_groot, chkpt_step=chkpt_step)

        # Linear comparators
        # Source: https://github.com/francois-rozet/piqa/blob/master/piqa/lpips.py
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(inplace=True) if use_dropout else nn.Identity(),
                nn.Conv2d(c, 1, kernel_size=1, bias=False),
            ) for c in self.vgg16_sliced.n_channels_list
        ])
        # Load checkpoint from Google Drive
        if chkpt_step:
            chkpt_filepath = self.vgg16_sliced.fetch_checkpoint(epoch_or_id=f'{chkpt_step}_linear', step=None)
            self.vgg16_sliced.logger.info(f'[VGG16] Loading {chkpt_filepath} in linear layers')
            lin_state_dict = torch.load(chkpt_filepath, map_location='cpu')
            self.linear_layers.load_state_dict(
                OrderedDict((re.sub(r'lin([0-9]).model.([0-9])', r'\1.\2', k), v)
                            for k, v in lin_state_dict.items())
            )
        # Freeze networks
        if not requires_grad:
            self.vgg16_sliced = self.vgg16_sliced.eval().requires_grad_(False)
            self.linear_layers = self.linear_layers.eval().requires_grad_(False)

    def forward_single(self, x: Tensor, reduction: str = 'mean') -> Tensor:
        """
        Compute LPIPS given a single batch of inputs
        :param (Tensor) x: input
        :param (str) reduction: one of 'mean', 'sum'
        :return:
        """
        # Extract features
        vgg_features = self.vgg16_sliced(x)
        # Normalize each feature vector to unit length over channel dimension.
        normalized_features = [xf / (xf.square().sum(dim=1, keepdims=True) ** 0.5 + 1e-10)
                               for xf in vgg_features]
        # Split and compute MSE
        feature_diff_mse = [torch.subtract(*nf.chunk(2, dim=0)).square()
                            for nf in normalized_features]
        feature_diff_flat = [self.linear_layers[i](fd_mse).mean(dim=(-1, -2), keepdim=True)
                             for i, fd_mse in enumerate(feature_diff_mse)]
        return torch.stack(feature_diff_flat, dim=0).sum(dim=0)

    def forward(self, x: Tensor, y: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
        """
        Compute LPIPS given a single batch of inputs
        :param (Tensor) x: input
        :param (Tensor) y: target
        :param (str) reduction: one of 'mean', 'sum'
        :return:
        """
        if y is None:
            return self.forward_single(x=x, reduction=reduction)

        # Extract features
        feat_x = self.vgg16_sliced(x)
        feat_y = self.vgg16_sliced(y)
        # Normalize each feature vector to unit length over channel dimension.
        norm_feat_x = [xf / (xf.square().sum(dim=1, keepdims=True) ** 0.5 + 1e-10)
                       for xf in feat_x]
        norm_feat_y = [yf / (yf.square().sum(dim=1, keepdims=True) ** 0.5 + 1e-10)
                       for yf in feat_y]
        # Split and compute MSE
        feat_diff = [torch.subtract(nfx, nfy).square()
                     for nfx, nfy in zip(norm_feat_x, norm_feat_y)]
        feat_diff_mse = [self.linear_layers[i](fd_mse).mean(dim=(-1, -2), keepdim=True)
                         for i, fd_mse in enumerate(feat_diff)]
        return torch.stack(feat_diff_mse, dim=0).sum(dim=0)


if __name__ == '__main__':
    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _log_level = 'debug'

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _fs = LocalFilesystem(LocalCapsule(_local_gdrive_root))
    _groot = LocalFolder.root(capsule_or_fs=_fs)

    # Define folder roots
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Initialize model
    # _vgg16 = VGG16(model_gfolder_or_groot=_models_groot, chkpt_step='397923af', crop_fc=False)
    # print(_vgg16)
    torch.cuda.empty_cache()
    _lpips_vgg = LPIPSLoss(model_gfolder_or_groot=_models_groot, chkpt_step='397923af')
    _lpips_vgg = _lpips_vgg.to('cuda')
    _x1 = torch.randn(3, 3, 224, 224, device='cuda')
    _lpips1 = _lpips_vgg(_x1)
    print(_lpips1.shape)

    torch.cuda.empty_cache()
    _x2 = torch.randn(3, 3, 224, 224, device='cuda')
    _lpips2 = _lpips_vgg(_x2)
    print(_lpips2.shape)
    print(((_lpips1 - _lpips2).square().sum(dim=-1) / 1e-4 ** 2).shape)

    torch.cuda.empty_cache()
    _lpips = _lpips_vgg(_x1, _x2)
    print(_lpips.shape)
    print(torch.mean(_lpips.squeeze()))
