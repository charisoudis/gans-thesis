import unittest
import torch
from modules.discriminators.patch_gan import PatchGANDiscriminator


class TestPatchGANDiscriminator(unittest.TestCase):

    def test_shapes(self):
        n_contracting_blocks = 4
        test_disc = PatchGANDiscriminator(10, 1, n_contracting_blocks=n_contracting_blocks)
        self.assertEqual(
            tuple(test_disc(
                torch.randn(1, 5, 256, 256),
                torch.randn(1, 5, 256, 256)
            ).shape),
            (1, 1, 256 // 2 ** n_contracting_blocks, 256 // 2 ** n_contracting_blocks)
        )

    def test_loss(self):
        pass
