from typing import Union, Sized

import numpy as np
import torch
from torch.utils.data import Sampler

from utils.command_line_logger import CommandLineLogger


class InfiniteSampler(Sampler):
    """
    InfiniteSampler Class:
    Sampler for torch.utils.data.DataLoader that loops over the dataset indefinitely, shuffling items as it goes.
    Source: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/misc.py
    """

    def __init__(self, data_source: Sized, shuffle: bool = True, seed: int = 42, rank: int = 0, num_replicas: int = 1,
                 window_size=0.5, logger: Union[CommandLineLogger, None] = None):
        self.data_len = len(data_source)
        assert self.data_len > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super(InfiniteSampler, self).__init__(data_source=data_source)
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.logger = logger
        assert self.logger is not None, 'Please provide a logger instance for InfiniteSampler'

    def __iter__(self):
        order = np.arange(self.data_len)
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

    def __len__(self):
        return np.inf


class ResumableRandomSampler(Sampler):
    """
    ResumableRandomSampler Class:
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    Original source: https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5
    """

    def __init__(self, data_source: Sized, shuffle: bool = True, seed: int = 42,
                 logger: Union[CommandLineLogger, None] = None):
        """
        ResumableRandomSampler class constructor.
        generator (Generator): Generator used in sampling.
        :param (Sized) data_source: torch.utils.data.Dataset or generally typings.Sized object of the dataset to draw
                                    samples from
        :param (int) seed: generator manual seed parameter
        :param (optional) logger: CommandLineLogger instance
        """
        super(ResumableRandomSampler, self).__init__(data_source=data_source)

        self.n_samples = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)

        self.shuffle = shuffle
        self.perm_index = 0
        if self.shuffle:
            self.perm = None
            self.reshuffle()
        else:
            self.perm = range(0, self.n_samples)

        self.logger = logger
        assert self.logger is not None, 'Please provide a logger instance for ResumableRandomSampler'

    def reshuffle(self) -> None:
        self.perm_index = 0
        if self.shuffle:
            self.perm = list(torch.randperm(self.n_samples, generator=self.generator).numpy())

    def __iter__(self):
        # If reached the end of dataset, reshuffle
        if self.perm_index >= len(self.perm):
            if self.logger:
                self.logger.debug(f'[SAMPLER] Reached end of epoch. Resetting state... (shuffle = {self.shuffle})')
            self.reshuffle()

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.n_samples

    def get_state(self) -> dict:
        return {
            "shuffle": self.shuffle,
            "perm": self.perm,
            "perm_index": self.perm_index,
            "generator_state": self.generator.get_state()
        }

    def set_state(self, state: dict) -> None:
        self.shuffle = bool(state.get("shuffle", True))
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])
