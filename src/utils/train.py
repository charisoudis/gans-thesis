from typing import Union

import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR


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
