import os
from typing import Any, Optional

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torchvision.transforms.functional as f
from torch.autograd import Function
from torch.nn import Module
from torchvision.transforms import transforms

from utils.command_line_logger import CommandLineLogger
from utils.ifaces import Verbosable
from utils.string import to_human_readable


def corr(x: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    """
    Compute the correlation matrix of given input vector $x$ using the unbiased sample correlation estimator.
    (credits to mauricett, https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393)
    :param x: the input vector of shape (N, n_vars)
    :param eps: constant for numerical stability when normalizing $x$
    :return: a torch.Tensor of shape (n_vars, n_vars)
    """
    n_samples, n_vars = tuple(x.shape)
    x -= torch.mean(x, dim=0)
    x /= torch.std(x, dim=0) + eps
    print('x.shape', x.shape)
    return 1 / (n_samples - 1) * x.transpose(-1, -2) @ x


def cov(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the covariance matrix of given input vector $x$ using the unbiased sample covariance estimator.
    (credits to mauricett, https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393)
    :param x: the input vector of shape (N, n_vars)
    :return: a torch.Tensor of shape (n_vars, n_vars)
    """
    n_samples, n_vars = tuple(x.shape)
    x -= torch.mean(x, dim=0)
    return 1 / (n_samples - 1) * x.transpose(-1, -2) @ x


def enable_verbose(model: nn.Module, logger: Optional[CommandLineLogger] = None) -> None:
    """
    Register verbose hooks on model's forward pass to output the shape of each layer's output tensor.
    :param (nn.Module) model: an torch.nn.Module instance
    :param (optional) logger: set internal logger
    """
    if logger is None:
        logger = model.logger if hasattr(model, 'logger') else \
            CommandLineLogger(log_level=os.getenv('TRAIN_LOG_LEVEL', 'debug'), name='ModelVerbose')
    # Recursively set for submodules
    if isinstance(model, Verbosable):
        for attr_name in model.get_layer_attr_names():
            enable_verbose(getattr(model, attr_name), logger=logger)
    else:
        # Register a hook for each layer in root module
        for _name, _layer in model.named_children():
            _layer.register_forward_hook(lambda _l, _, _out: logger.debug(f"{_name}: {_out.shape}"))
    # Flag model
    model.verbose_enabled = True


def get_gradient(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient of the critic's scores w.r.t. mixes of real and fake images.
    Attention: Returns the gradient of c(x) w.r.t. to x. If for example x is a (3,128,128) image, then grad_x(c(x)) will
               be a (3,128,128) tensor containing how much the c(x) changes if x[i][j][k] changes.
    :param (nn.Module) disc: the critic model
    :param (torch.Tensor) real: a batch of real images
    :param (torch.Tensor) fake: a batch of fake images
    :param (torch.Tensor) epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    :return: (torch.Tensor) the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images)

    # Take the gradient of the scores with respect to the images
    return torch.autograd.grad(
        # Note: We need to take the gradient of outputs with respect to inputs.
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with how the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]


def get_gradient_penalty_from_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, we calculate the magnitude of each image's gradient and penalize the mean
    quadratic distance of each magnitude to 1.
    :param (torch.Tensor) gradient: the gradient of the critic's scores, with respect to the mixed image
    :return: a scalar torch.Tensor object containing the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    return torch.mean((torch.ones_like(gradient_norm) - gradient_norm) ** 2)


def get_gradient_penalty(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor) \
        -> torch.Tensor:
    """
    Get the gradient penalty regularization term, given the discriminator (critic) model, a set of real and fake and
    images and the parameter :attr:`epsilon` to mix the two set of images (approximating an average input to critic).
    :param (nn.Module) disc: the critic model
    :param (torch.Tensor) real: a batch of real images
    :param (torch.Tensor) fake: a batch of fake images
    :param (torch.Tensor) epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    :return: (torch.Tensor) the gradient of the critic's scores, with respect to the mixed image
    """
    return get_gradient_penalty_from_gradient(
        get_gradient(disc=disc, real=real, fake=fake, epsilon=epsilon)
    )


def get_total_params(model: Module, print_table: bool = False, sort_desc: bool = False) -> int or None:
    """
    Get total number of parameters from given nn.Module.
    :param model: model to count parameters for
    :param print_table: if True prints counts for every sub-module and returns None, else returns total count only
    :param sort_desc: if True sorts array in DESC order wrt to parameter count before printing
    :return: total number of parameters if $print$ is set to False, else prints counts and returns nothing
    """
    total_count_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = 0
    count_dict = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        count_dict.append({
            'name': name,
            'count': count,
            'count_hr': to_human_readable(count),
            'prc': '%.2f %%' % (count / total_count_orig)
        })
        total_count += count

    assert total_count_orig == total_count, "Should be equal..."

    if print_table is True:
        from json import dumps
        from prettytable import from_json

        if sort_desc:
            count_dict = sorted(count_dict, key=lambda k: k['count'], reverse=True)

        table = from_json(dumps(count_dict, indent=4))
        table.field_names = ["Module", "Count", "Count Human", "Percentage"]

        print(table.get_string(fields=["Module", "Count Human", "Percentage"]))
        print(f"Total Trainable Params: {to_human_readable(total_count)}")
        return None

    return total_count


def invert_transforms(ts: transforms.Compose) -> transforms.Compose:
    """
    Inverts Normalize and ToTensor transforms in a transformation sequence defined via torchvision.transforms.Compose.
    :param ts: transforms.Compose object with the list of transforms to be inverted
    :return: a torchvision.transforms.Compose object with the last transforms inverted
    """
    inverse_transforms_list = []
    for i in reversed(range(len(ts.transforms))):
        t = ts.transforms[i]
        if type(t) == transforms.Normalize:
            inverse_transforms_list.append(UnNormalize(mean=t.mean, std=t.std, inplace=t.inplace))
        elif type(t) == transforms.ToTensor:
            # inverse_transforms_list.append(transforms.ToPILImage())
            # We reached the ToTensor() transform; no point in continuing inverting upwards where the image was still a
            # PIL image object
            break
    return transforms.Compose(inverse_transforms_list)


def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of the given matrix.
    :param mat: the input matrix as a torch.Tensor object
    :return: a torch.Tensor fo the same shape as the input
    """
    return MatrixSquareRoot.apply(mat)


class MatrixSquareRoot(Function):
    """
    MatrixSquareRoot Class:
    This class is used to compute square root of a positive definite matrix given as torch.Tensor object.
    NOTE: matrix square root is not differentiable for matrices with zero eigenvalues.
    Source: https://github.com/steveli/pytorch-sqrtm
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx: Any, mat: torch.Tensor, **kwargs):
        mat_np = mat.detach().cpu().numpy().astype(np.float_)
        mat_sqrt = scipy.linalg.sqrtm(mat_np).real
        mat_sqrt = torch.from_numpy(mat_sqrt).to(mat)
        ctx.save_for_backward(mat_sqrt)
        return mat_sqrt

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        grad_input = None
        if ctx.needs_input_grad[0]:
            mat_sqrt, = ctx.saved_tensors
            mat_sqrt = mat_sqrt.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_mat_sqrt = scipy.linalg.solve_sylvester(mat_sqrt, mat_sqrt, gm)
            grad_input = torch.from_numpy(grad_mat_sqrt).to(grad_output)
        return grad_input


class ToTensorOrPass(transforms.ToTensor):
    """
    ToTensorOrPass Class:
    Completes ToTensor() transform class by skipping if input is already a tensor.
    """

    def __init__(self, renormalize: bool = True):
        """
        ToTensorOrPass class constructor.
        :param (bool) renormalize: set to True to renormalize input tensors to [0,1], otherwise if Tensor object
                                   encountered in input it will return it intact
        """
        self.renormalize = renormalize

    def __call__(self, pic_or_tensor):
        # case: PIL image
        if type(pic_or_tensor) is not torch.Tensor:
            return super(ToTensorOrPass, self).__call__(pic_or_tensor)

        # case: torch.Tensor
        if self.renormalize:
            tensor_min = torch.min(pic_or_tensor)
            tensor_max = torch.max(pic_or_tensor)
            return (pic_or_tensor - tensor_min) / (tensor_max - tensor_min)

        return pic_or_tensor


class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = f.normalize(tensor, list(np.zeros_like(self.mean)), list(np.reciprocal(self.std)), self.inplace)
        return f.normalize(tensor, list(-1 * np.asarray(self.mean)), list(np.ones_like(self.std)), self.inplace)
