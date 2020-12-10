import unittest
import torch

from utils.pytorch import get_total_params, to_human_readable


class TestPytorchUtils(unittest.TestCase):

    def test_to_human_readable(self) -> None:
        number = 11
        self.assertEqual('11', to_human_readable(number))
        number = 1_000
        self.assertEqual('1K', to_human_readable(number))
        number = 1_435
        self.assertEqual('1.4K', to_human_readable(number))
        number = 1_000_000
        self.assertEqual('1M', to_human_readable(number))
        number = 1_500_000
        self.assertEqual('1.5M', to_human_readable(number))
        number = 1_500_500
        self.assertEqual('1.5M', to_human_readable(number))
        number = 1_505_500
        self.assertEqual('1.51M', to_human_readable(number, size_format='%.2f'))
        number = 1_515_500_001
        self.assertEqual('1.52B', to_human_readable(number, size_format='%.2f'))

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
