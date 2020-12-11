import unittest
import torch
from modules.generators.munit import MUNITGenerator


class TestMUNIT(unittest.TestCase):

    def setUp(self) -> None:
        self.s_dim = 8
        self.c_enc_n_contracting_blocks = 2
        self.s_enc_n_contracting_blocks = 4
        self.c_base = 64
        self.test_gen = MUNITGenerator(s_dim=self.s_dim, n_content_downsampling_blocks=self.c_enc_n_contracting_blocks,
                                       n_style_downsampling_blocks=self.s_enc_n_contracting_blocks, c_base=self.c_base)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self) -> None:
        x_content_code, x_style_code = self.test_gen.encode(self.x)
        temp = 2 ** self.c_enc_n_contracting_blocks
        self.assertEqual(tuple(x_content_code.shape), (1, self.c_base * temp, 256 // temp, 256 // temp))
        self.assertEqual(tuple(x_style_code.shape), (1, self.s_dim, 1, 1))

        x_hat = self.test_gen.decode(x_content_code, x_style_code)
        self.assertEqual(tuple(x_hat.shape), tuple(self.x.shape))

    def test_losses_can_run(self) -> None:
        ae_loss, c, s = self.test_gen.ae_image_recon_loss(self.x)
        self.assertEqual(tuple(ae_loss.shape), ())
        self.assertEqual(len(c), 1)
        self.assertEqual(tuple(s.shape), (1, self.s_dim, 1, 1))

        content_recon_loss, style_recon_loss, x_hat = self.test_gen.ae_latent_recon_loss(c, s)
        self.assertEqual(tuple(content_recon_loss.shape), ())
        self.assertEqual(tuple(style_recon_loss.shape), ())
        self.assertEqual(tuple(x_hat.shape), tuple(self.x.shape))

