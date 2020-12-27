import os
import sys
from typing import Union, Optional, Sized, Tuple

import numpy as np
import torch
from IPython import get_ipython
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
# noinspection PyProtectedMember
from torch.utils.data import Sampler, random_split, Dataset


class ResumableRandomSampler(Sampler):
    """
    ResumableRandomSampler Class:
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    Original source: https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5
    """

    def __init__(self, data_source: Sized, shuffle: bool = True, seed: int = 42):
        """
        ResumableRandomSampler class constructor.
        generator (Generator): Generator used in sampling.
        :param data_source: torch.utils.data.Dataset or generally typings.Sized object of the dataset to sample from
        :param seed: generator manual seed parameter
        """
        super(ResumableRandomSampler, self).__init__(data_source=data_source)

        self.n_samples = len(data_source)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.shuffle = shuffle
        if self.shuffle:
            self.perm_index = None
            self.perm = None
            self.reshuffle()
        else:
            self.perm_index = 0
            self.perm = range(0, self.n_samples)

    def reshuffle(self) -> None:
        self.perm_index = 0
        self.perm = list(torch.randperm(self.n_samples, generator=self.generator).numpy())

    def __iter__(self):
        # If reached the end of dataset, reshuffle
        if self.perm_index >= len(self.perm):
            if self.shuffle:
                self.reshuffle()
            else:
                self.perm_index = 0

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.n_samples

    def get_state(self) -> dict:
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state: dict) -> None:
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


def load_model_chkpt(model: nn.Module, model_name: str, dict_key: Optional[str] = None,
                     model_opt: Optional[Optimizer] = None, chkpts_root: Optional[dict] = None,
                     state_dict: Optional[dict] = None) -> Tuple[Optional[int], dict]:
    """
    Load model (and model's optimizer) checkpoint. The checkpoint is searched in given checkpoints root (absolute path)
    and if one found it is loaded. The function also returns the checkpoint step as well as
    :param (nn.Module) model: the model as a torch.nn.Module instance
    :param (str) model_name: name of model which is also model checkpoint's file name prefix
    :param (optional) dict_key: name of the key state dictionary regarding the model's state
    :param (optional) model_opt: model's optimizer instance
    :param (optional) chkpts_root: absolute path to model checkpoints directory
    :param (optional) state_dict: state dict (used to avoid duplicate calls)
    :return: a tuple containing the total number of images and the loaded state dict
    """
    # Check if running inside Colab or Kaggle (auto prefixing)
    if 'google.colab' in sys.modules or 'google.colab' in str(get_ipython()) or 'COLAB_GPU' in os.environ:
        chkpts_root = f'/content/drive/MyDrive/Model Checkpoints'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        chkpts_root = f'/kaggle/working/Model Checkpoints'
    elif not chkpts_root:
        chkpts_root: str = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints'
    assert os.path.exists(chkpts_root) and os.path.isdir(chkpts_root), 'Checkpoints dir not existent or not readable'
    assert model_opt is None or dict_key is not None, 'model_opt and dict_key cannot be None simultaneously'

    if not state_dict:
        # Find correct checkpoint path
        _, _, chkpt_files = next(os.walk(chkpts_root))
        chkpt_files = sorted([_f for _f in chkpt_files if _f.lower().startswith(model_name.lower())], reverse=True)
        assert len(chkpt_files) > 0, 'No model checkpoints found in given checkpoints dir'
        chkpt_file = chkpt_files[0]
        chkpt_info_parts = chkpt_file.replace(model_name, '').lstrip('_').replace('.pth', '').split('_')
        chkpt_path = os.path.join(chkpts_root, chkpt_file)

        # Load checkpoint
        state_dict = torch.load(chkpt_path, map_location='cpu')
    else:
        chkpt_info_parts = ['None']

    assert dict_key is None or dict_key in state_dict.keys(), f'dict_key={str(dict_key)} not found in state_dict.keys()'
    model.load_state_dict(state_dict[dict_key] if dict_key else state_dict)
    if model_opt:
        assert f'{dict_key}_opt' in state_dict.keys(), '$dict_key$_opt not found in state_dict.keys()'
        model_opt.load_state_dict(state_dict[f'{dict_key}_opt'])

    # Find epoch/current step
    chkpt_images = None
    if len(chkpt_info_parts) == 2:
        chkpt_cur_step = int(chkpt_info_parts[0])
        chkpt_batch_size = int(chkpt_info_parts[1])
        chkpt_images = chkpt_cur_step * chkpt_batch_size
    elif len(chkpt_info_parts) == 0 and chkpt_info_parts[0].isnumeric():
        chkpt_images = int(chkpt_info_parts[0])
    return chkpt_images, state_dict


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


def get_optimizer_lr_scheduler(optimizer: Optimizer, schedule_type: str, **kwargs) \
        -> Union[CyclicLR, ReduceLROnPlateau]:
    """
    Set optimiser's learning rate scheduler based on $schedule_type$ string.
    :param optimizer: instance of torch.optim.Optimizer subclass
    :param schedule_type: learning-rate schedule type (supported: 'on_plateau', 'cyclic',)
    :param kwargs: scheduler-specific keyword arguments
    """
    switcher = {
        'on_plateau': ReduceLROnPlateau,
        'cyclic': CyclicLR,
    }
    return switcher[schedule_type](optimizer=optimizer, **kwargs)


def train_test_split(dataset: Union[Dataset, Sized], splits: list, seed: int = 42) \
        -> Tuple[Union[Dataset, Sized], Union[Dataset, Sized]]:
    """
    Split :attr:`dataset` to training set and test set based on the percentages from :attr:`splits`.
    :param dataset: the dataset upon which to perform the split
    :param splits: the percentages of the split, (training_set, test_set) (e.g (90, 10) or (0.9, 0.1), both acceptable)
    :param seed: the manual seed parameter of the split
    :return: a tuple containing the two subsets as torch.utils.data.Dataset objects
    """
    # Get splits
    dataset_len = len(dataset)
    splits = np.array(splits, dtype=np.float32)
    if splits[0] > 1:
        splits *= 0.01
    splits *= dataset_len
    split_lengths = np.floor(splits).astype(np.int32)
    split_lengths[0] += dataset_len - split_lengths.sum()
    # Perform the split
    train_set, test_set = random_split(dataset, lengths=split_lengths, generator=torch.Generator().manual_seed(seed))
    return train_set, test_set


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
