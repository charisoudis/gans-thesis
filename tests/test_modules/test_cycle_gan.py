import unittest
import torch

from modules.cycle_gan import CycleGAN


class TestCycleGAN(unittest.TestCase):

    def setUp(self) -> None:
        self.test_cycle_gan = CycleGAN(c_in=1, c_out=1, lr_scheduler_type='on_plateau')
        self.test_real_a = torch.randn(1, 1, 256, 256)
        self.test_real_b = torch.randn(1, 1, 256, 256)

    def test_forward(self) -> None:
        disc_a_loss, disc_b_loss, gen_loss, fake_a, fake_b = \
            self.test_cycle_gan(real_a=self.test_real_a, real_b=self.test_real_b)
        self.assertEqual(tuple(fake_a.shape), tuple(self.test_real_a.shape))
        self.assertEqual(tuple(fake_b.shape), tuple(self.test_real_b.shape))
        self.assertEqual(tuple(disc_a_loss.shape), ())
        self.assertEqual(tuple(disc_b_loss.shape), ())
        self.assertEqual(tuple(gen_loss.shape), ())
