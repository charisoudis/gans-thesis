import unittest
import torch
import torch.nn as nn

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.pix2pix import UNet


class TestPix2PixGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.c_out = 3
        self.test_unet = UNet(c_in=3, c_out=self.c_out)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self) -> None:
        self.assertEqual(tuple(self.test_unet(self.x).shape), (1, self.c_out, 256, 256))

    def test_losses_can_run(self) -> None:
        test_disc = PatchGANDiscriminator(c_in=3 + self.c_out)
        loss = self.test_unet.get_loss(test_disc, real=torch.randn_like(self.x), condition=self.x,
                                       adv_criterion=nn.BCELoss(), recon_criterion=nn.L1Loss(), lambda_recon=2)
        self.assertEqual(tuple(loss.shape), ())
