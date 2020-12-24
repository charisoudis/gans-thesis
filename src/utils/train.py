import os
import sys
from typing import Union, Optional

import torch
from IPython import get_ipython
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR


def load_model_chkpt(model: nn.Module, model_name: str, dict_key: Optional[str] = None,
                     model_opt: Optional[Optimizer] = None,
                     chkpts_root: str = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints') -> Optional[int]:
    """
    Load model (and model's optimizer) checkpoint. The checkpoint is searched in given checkpoints root (absolute path)
    and if one found it is loaded. The function also returns the checkpoint step as well as
    :param (nn.Module) model: the model as a torch.nn.Module instance
    :param (str) model_name: name of model which is also model checkpoint's file name prefix
    :param (Optimizer) model_opt: (optional) model's optimizer instance
    :param (str) dict_key: (optional) name of the key state dictionary regarding the model's state
    :param (str) chkpts_root: absolute path to model checkpoints directory
    :return: the total number of images
    """
    # If run from inside Google Colab, then override given path
    inside_colab = 'google.colab' in sys.modules or \
                   'google.colab' in str(get_ipython()) or \
                   'COLAB_GPU' in os.environ
    if inside_colab:
        chkpts_root = '/content/drive/MyDrive/Model Checkpoints'
    assert os.path.exists(chkpts_root) and os.path.isdir(chkpts_root), 'Checkpoints dir not existent or not readable'
    assert model_opt is None or dict_key is not None, 'model_opt and dict_key cannot be None simultaneously'

    # Find correct checkpoint path
    _, _, chkpt_files = next(os.walk(chkpts_root))
    chkpt_files = sorted([_f for _f in chkpt_files if _f.lower().startswith(model_name.lower())], reverse=False)
    assert len(chkpt_files) > 0, 'No model checkpoints found in given checkpoints dir'
    chkpt_file = chkpt_files[0]
    chkpt_path = os.path.join(chkpts_root, chkpt_file)

    # Load checkpoint
    state_dict = torch.load(chkpt_path, map_location='cpu')
    assert dict_key is None or dict_key in state_dict.keys(), f'dict_key={str(dict_key)} not found in state_dict.keys()'
    model.load_state_dict(state_dict[dict_key] if dict_key else state_dict)
    if model_opt:
        assert f'{dict_key}_opt' in state_dict.keys(), '$dict_key$_opt not found in state_dict.keys()'
        model_opt.load_state_dict(state_dict[f'{dict_key}_opt'])

    # Find epoch/current step
    chkpt_info_parts = chkpt_file.replace(model_name, '').lstrip('_').replace('.pth', '').split('_')
    chkpt_images = None
    if len(chkpt_info_parts) == 2:
        chkpt_cur_step = int(chkpt_info_parts[0])
        chkpt_batch_size = int(chkpt_info_parts[1])
        chkpt_images = chkpt_cur_step * chkpt_batch_size
    elif len(chkpt_info_parts) == 0 and chkpt_info_parts[0].isnumeric():
        chkpt_images = int(chkpt_info_parts[0])
    return chkpt_images


def get_adam_optimizer(*models, lr: float = 1e-4, betas: tuple = (0.9, 0.999), delta: float = 1e-8) -> Optimizer:
    """
    Get Adam optimizer for jointly training ${models} argument.
    :param models: one or more models to apply optimizer on
    :param lr: learning rate
    :param betas: (p_1, p_2) exponential decay parameters for Adam optimiser's moment estimates. According to
    Goodfellow, good defaults are (0.9, 0.999).
    :param delta: small constant that's used for numerical stability. According to Goodfellow, good defaults is 1e-8.
    :return: instance of torch.optim.Adam optimizer
    """
    joint_params = []
    for model in models:
        joint_params += list(model.parameters())
    return Adam(joint_params, lr=lr, betas=betas, eps=delta)


def set_optimizer_lr(optimizer: Optimizer, new_lr: float) -> None:
    """
    Set optimiser's learning rate to match $new_lr$.
    :param optimizer: instance of torch.optim.Optimizer subclass
    :param new_lr: the new learning rate as a float
    """
    for group in optimizer.param_groups:
        group['lr'] = new_lr


def get_optimizer_lr_scheduler(optimizer: Optimizer, schedule_type: str, *args) -> Union[CyclicLR, ReduceLROnPlateau]:
    """
    Set optimiser's learning rate scheduler based on $schedule_type$ string.
    :param optimizer: instance of torch.optim.Optimizer subclass
    :param schedule_type: learning-rate schedule type (supported: 'on_plateau', 'cyclic',)
    :param args: scheduler argument list
    """
    switcher = {
        'on_plateau': ReduceLROnPlateau,
        'cyclic': CyclicLR,
    }
    return switcher[schedule_type](optimizer=optimizer, *args)


def weights_init_naive(module: nn.Module) -> None:
    """
    Apply naive weight initialization in given nn.Module. Should be called like network.apply(weights_init_naive).
    This naive approach simply sets all biases to 0 and all weights to the output of normal distribution with mean of 0
    and a std of 5e-2.
    :param module: input module
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.normal_(module.weight, 0., 5.0e-2)
    if isinstance(module, nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight, 0., 5.0e-2)
        torch.nn.init.constant_(module.bias, 0.)
