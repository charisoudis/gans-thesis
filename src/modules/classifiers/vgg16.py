import os
from collections import namedtuple
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from torchvision.models import vgg16

from modules.ifaces import IGModule
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.ifaces import FilesystemFolder


class VGG16(nn.Module, IGModule):
    """
    VGG16 Class:
    This class is used to access and use VGG16 nn.Module but with the additional functionality provided from
    inheriting `utils.gdrive.GDriveModel`. Inheriting GDriveModel we can easily download / upload model checkpoints to
    Google Drive using GoogleDrive API's python client.
    """

    DefaultConfiguration = {'crop_fc': False}

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

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param x: the input batch of images as a `torch.Tensor` object
        :return: a tuple containing the intermediate feature activations as `torch.Tensor` objects
        """
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
    _inception = VGG16(model_gfolder_or_groot=_models_groot, chkpt_step='397923af', crop_fc=False)
    print(_inception)
