import tqdm
import tqdm.notebook as tqdm_nb


# noinspection PyUnresolvedReferences
def get_tqdm() -> Type[Union[tqdm, tqdm_nb]]:
    """
    Get the correct Tqdm instance for showing progress. This is due to the fact that `tqdm.tqdm` is not working
    correctly in IPython notebook.
    :return: either an `tqdm.tqdm` or an `tqdm.notebook.tqdm` instance depending on execution context
    """
    try:
        _ = __IPYTHON__
        return tqdm_nb
    except NameError:
        return tqdm
