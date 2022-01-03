import math
from typing import Type, Union

from IPython import get_ipython
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb


def closest_pow(x: int or float, of: int = 2) -> int or float:
    """
    Compute closest power of :attr:`of` for given number :attr:`x`
    :param (int or float) x:
    :param int of: exponent
    :return: the same data type as :attr:`x`
    """
    return type(x)(pow(of, round(math.log(x) / math.log(of))))


def in_notebook() -> bool:
    """
    Checks if code is run from inside
    :return:
    """
    try:
        shell_class = get_ipython().__class__
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or 'google.colab' in str(shell_class):
            return True  # Jupyter notebook or Qt console
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# noinspection PyUnresolvedReferences
def get_tqdm() -> Type[Union[tqdm_nb, tqdm]]:
    """
    Get the correct Tqdm instance for showing progress. This is due to the fact that `tqdm.tqdm` is not working
    correctly in IPython notebook.
    :return: either an `tqdm.tqdm` or an `tqdm.notebook.tqdm` instance depending on execution context
    """
    return tqdm_nb if in_notebook() else tqdm
