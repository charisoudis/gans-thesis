import unittest
import torch
from modules.generators.pix2pix import UNet


class TestPix2PixGenerator(unittest.TestCase):

    def setUp(self):
        self.c_out = 10
        self.test_unet = UNet(c_in=3, c_out=self.c_out)
        self.x = torch.rand(1, 3, 256, 256)

    def test_shapes(self):
        self.assertEqual(tuple(self.test_unet(self.x).shape), (1, self.c_out, 256, 256))

    def test_losses(self):
        # TODO: when added pix2pix generator losses, fill this unit test
        pass
