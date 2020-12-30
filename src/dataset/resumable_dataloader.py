import abc

# noinspection PyProtectedMember
from torch.utils.data import DataLoader


class ResumableDataLoader(DataLoader):

    def __init__(self, **kwargs):
        super(ResumableDataLoader, self).__init__(**kwargs)

    @abc.abstractmethod
    def get_state(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: dict) -> None:
        raise NotImplementedError
