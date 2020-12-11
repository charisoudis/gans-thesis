import unittest
import torch

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator


class TestCycleGANGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.test_gen = CycleGANGenerator(c_in=3, c_out=3)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self) -> None:
        self.assertEqual(tuple(self.test_gen(self.x).shape), tuple(self.x.shape))

    def test_if_tanh(self) -> None:
        x_hat = self.test_gen(self.x)
        self.assertLessEqual(x_hat.max().item(), 1.0)
        self.assertGreaterEqual(x_hat.min().item(), -1.0)

    def test_losses_can_run(self) -> None:
        test_disc_y = PatchGANDiscriminator(c_in=3)
        adv_loss, fake_y = self.test_gen.get_adv_loss(real_x=self.x, disc_y=test_disc_y)
        self.assertEqual(tuple(adv_loss.shape), ())
        self.assertEqual(tuple(fake_y.shape), tuple(self.x.shape))

        identity_loss, identity_y = self.test_gen.get_identity_loss(real_y=self.x)
        self.assertEqual(tuple(identity_loss.shape), ())
        self.assertEqual(tuple(identity_y.shape), tuple(self.x.shape))

        cycle_consistency_loss, cycle_y = self.test_gen.get_cycle_consistency_loss(fake_x=self.x, real_y=self.x)
        self.assertEqual(tuple(cycle_consistency_loss.shape), ())
        self.assertEqual(tuple(cycle_y.shape), tuple(self.x.shape))

