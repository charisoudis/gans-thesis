import unittest

import torch

from utils.pytorch import get_total_params


class TestPytorchUtils(unittest.TestCase):

    def test_get_total_params(self) -> None:
        conf = {
            'c_in': 3,
            'c_out': 10,
            'kernel': 3,
            'bias': True
        }
        test_module = torch.nn.Conv2d(in_channels=conf['c_in'], out_channels=conf['c_out'], kernel_size=conf['kernel'],
                                      bias=conf['bias'])
        self.assertEqual(conf['c_out'] * (conf['c_in'] * conf['kernel'] ** 2 + (1 if conf['bias'] else 0)),
                         get_total_params(test_module))
