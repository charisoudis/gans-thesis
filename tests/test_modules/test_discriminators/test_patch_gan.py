import unittest
import torch
from modules.discriminators.patch_gan import PatchGANDiscriminator


class TestPatchGANDiscriminator(unittest.TestCase):

    def setUp(self) -> None:
        self.n_contracting_blocks = 4
        self.c_in = 6
        self.test_disc = PatchGANDiscriminator(self.c_in, n_contracting_blocks=self.n_contracting_blocks)

    def test_shapes(self) -> None:
        self.assertEqual(
            tuple(self.test_disc(
                torch.randn(1, self.c_in // 2, 256, 256),
                torch.randn(1, self.c_in // 2, 256, 256)
            ).shape),
            (1, 1, 256 // 2 ** self.n_contracting_blocks, 256 // 2 ** self.n_contracting_blocks)
        )

    def test_loss_can_run(self) -> None:
        real = torch.randn(1, self.c_in // 2, 256, 256)
        fake = torch.randn(1, self.c_in // 2, 256, 256)
        condition = torch.ones(1, self.c_in // 2, 256, 256)
        adv_loss = self.test_disc.get_loss(real=real, fake=fake, condition=condition)
        self.assertEqual(tuple(adv_loss.shape), ())
