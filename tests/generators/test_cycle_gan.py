import unittest
import torch
from modules.generators.cycle_gan import CycleGANGenerator


class TestCycleGANGenerator(unittest.TestCase):

    def setUp(self):
        self.test_gen = CycleGANGenerator(c_in=3, c_out=3)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self):
        self.assertEqual(tuple(self.test_gen(self.x).shape), tuple(self.x.shape))

    def test_if_tanh(self):
        x_hat = self.test_gen(self.x)
        self.assertLessEqual(x_hat.max().item(), 1.0)
        self.assertGreaterEqual(x_hat.min().item(), -1.0)

    def test_losses(self):
        # TODO: when added CycleGAN generator losses, fill this unit test
        pass
