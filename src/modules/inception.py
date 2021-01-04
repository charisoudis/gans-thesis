import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import inception_v3

from utils.gdrive import GDriveModel, GDriveFolder


class InceptionV3(nn.Module, GDriveModel):
    """
    InceptionV3 Class:
    TODO
    """

    ModelName = 'inceptionv3'

    def __init__(self, model_gfolder_or_groot: GDriveFolder, crop_fc: bool = False):
        """
        InceptionV3 class constructor.
        :param (GDriveFolder) model_gfolder_or_groot: a `utils.gdrive.GDriveFolder` object to download/upload model
                                                      checkpoints and metrics from/to Google Drive
        :param (bool) crop_fc: set to True to crop FC layer from Inception v3 network
        """
        # Instantiate GDriveModel class
        model_gfolder = model_gfolder_or_groot if model_gfolder_or_groot.name.endswith(self.ModelName) else \
            model_gfolder_or_groot.subfolder_by_name(folder_name=f'model_name={self.ModelName}', recursive=True)
        GDriveModel.__init__(self, model_gfolder=model_gfolder)
        # Instantiate InceptionV3 model
        nn.Module.__init__(self)
        self.inception_v3 = inception_v3(pretrained=False, init_weights=False)

        # Load checkpoint from Google Drive
        chkpt_filepath = self.fetch_latest_checkpoint()
        self.inception_v3.load_state_dict(torch.load(chkpt_filepath, map_location='cpu'))

        # Cutoff FC layer from Inception model when we do not want classification, but feature embedding
        if crop_fc:
            self.inception_v3.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        This method implements the forward pass through Inception v3 network.
        :param x: the input batch of images as a `torch.Tensor` object
        :return: a `torch.Tensor` object with batch of classifications
        """
        return self.inception_v3(x)
