from typing import Any

import numpy as np
import scipy.optimize
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module

from utils.string import to_human_readable


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


class MatrixSquareRoot(Function):
    """
    MatrixSquareRoot Class:
    This class is used to compute square root of a positive definite matrix given as torch.Tensor object.
    NOTE: matrix square root is not differentiable for matrices with zero eigenvalues.
    Source: https://github.com/steveli/pytorch-sqrtm
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx: Any, mat: Tensor, **kwargs):
        mat_np = mat.detach().cpu().numpy().astype(np.float_)
        mat_sqrt = scipy.linalg.sqrtm(mat_np).real
        mat_sqrt = torch.from_numpy(mat_sqrt).to(mat)
        ctx.save_for_backward(mat_sqrt)
        return mat_sqrt

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
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


def matrix_sqrt(mat: Tensor) -> Tensor:
    """
    Compute the square root of the given matrix.
    :param mat: the input matrix as a torch.Tensor object
    :return: a torch.Tensor fo the same shape as the input
    """
    return MatrixSquareRoot.apply(mat)


def cov(x: Tensor) -> Tensor:
    """
    Compute the covariance matrix of given input vector $x$ using the unbiased sample covariance estimator.
    (credits to mauricett, https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393)
    :param x: the input vector of shape (N, n_vars)
    :return: a torch.Tensor of shape (n_vars, n_vars)
    """
    n_samples, n_vars = tuple(x.shape)
    x -= torch.mean(x, dim=0)
    return 1 / (n_samples - 1) * x.transpose(-1, -2) @ x


def corr(x: Tensor, eps: float = 1e-08) -> Tensor:
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
