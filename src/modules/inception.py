from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import inception_v3

from utils.gdrive import GDriveModel, GDriveFolder
from utils.ifaces import Configurable


class InceptionV3(nn.Module, GDriveModel, Configurable):
    """
    InceptionV3 Class:
    This class is used to access and use Inception V3 nn.Module but with the additional functionality provided from
    inheriting `utils.gdrive.GDriveModel`. Inheriting GDriveModel we can easily download / upload model checkpoints to
    Google Drive using GoogleDrive API's python client.
    """

    def __init__(self, model_gfolder_or_groot: GDriveFolder, chkpt_step: Optional[int or str] = None,
                 crop_fc: bool = False):
        """
        InceptionV3 class constructor.
        :param (GDriveFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                      checkpoints and metrics from/to Google Drive
        :param (str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will be
                                         loaded via `nn.Module().load_state_dict()`
        :param (bool) crop_fc: set to True to crop FC layer from Inception v3 network (e.g. to get image embeddings)
        """
        # Instantiate GDriveModel class
        model_name = self.__class__.__name__.lower()
        model_gfolder = model_gfolder_or_groot if model_gfolder_or_groot.name.endswith(model_name) else \
            model_gfolder_or_groot.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=True)
        GDriveModel.__init__(self, model_gfolder=model_gfolder, model_name=model_name)
        # Instantiate InceptionV3 model
        nn.Module.__init__(self)
        self.inception_v3 = inception_v3(pretrained=False, init_weights=False)

        # Load checkpoint from Google Drive
        if chkpt_step:
            chkpt_filepath = self.fetch_checkpoint(step=chkpt_step)
            self.load_state_dict(torch.load(chkpt_filepath, map_location='cpu'))

        # Cutoff FC layer from Inception model when we do not want classification, but feature embedding
        if crop_fc:
            self.inception_v3.fc = nn.Identity()
        self.crop_fc = crop_fc

    def state_dict(self, *args, **kwargs) -> Dict[str, Tensor]:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        return self.inception_v3.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        return self.inception_v3.load_state_dict(state_dict=state_dict, strict=strict)

    def configuration(self) -> dict:
        return {'crop_fc': self.crop_fc}

    def load_configuration(self, configuration: dict) -> None:
        self.crop_fc = configuration['crop_fc']
        if self.crop_fc:
            self.inception_v3.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        This method implements the forward pass through Inception v3 network.
        :param x: the input batch of images as a `torch.Tensor` object
        :return: a `torch.Tensor` object with batch of classifications
        """
        # Update gdrive model state
        self.gforward(x.shape[0])
        # Perform the actual forward pass
        return self.inception_v3(x)
