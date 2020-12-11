import unittest
import torch
from modules.discriminators.munit import MUNITDiscriminator


class TestMUNITDiscriminator(unittest.TestCase):

    def test_shapes(self) -> None:
        n_blocks = 4
        w_in = 512

        test_disc = MUNITDiscriminator(n_discriminators=3, c_in=5, c_hidden=1, n_contracting_blocks=n_blocks)
        x = torch.randn(1, 5, w_in, w_in)
        predictions = test_disc(x)
        # Test correct number of discriminators
        self.assertEqual(len(predictions), 3)
        # Test correct number of contracting blocks
        biggest_output_dim = w_in // 2 ** n_blocks
        self.assertEqual(predictions[0].shape[-1], biggest_output_dim)
        # Test output for each of the discriminators
        for i, disc_pred in list(enumerate(predictions)):
            self.assertEqual(tuple(disc_pred.shape), (1, 1, biggest_output_dim // 2 ** i, biggest_output_dim // 2 ** i))
