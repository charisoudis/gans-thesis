import unittest

import torch
import torch.nn as nn

from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.unet import UNETWithSkipConnections


class TestUNETWithSkipConnections(unittest.TestCase):

    def setUp(self) -> None:
        self.c_out = 3
        self.test_unet = UNETWithSkipConnections(c_in=3, c_out=self.c_out)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self) -> None:
        self.assertEqual(tuple(self.test_unet(self.x).shape), (1, self.c_out, 256, 256))

        test_unet_fc = UNETWithSkipConnections(c_in=3, c_out=3, fc_in_bottleneck=True, w_in=256, h_in=256)
        self.assertEqual(tuple(test_unet_fc(self.x).shape), tuple(self.x.shape))

    def test_losses_can_run(self) -> None:
        test_disc = PatchGANDiscriminator(c_in=3 + self.c_out)
        loss = self.test_unet.get_loss(test_disc, real=torch.randn_like(self.x), condition=self.x,
                                       adv_criterion=nn.BCELoss(), recon_criterion=nn.L1Loss(), lambda_recon=2)
        self.assertEqual(tuple(loss.shape), ())
