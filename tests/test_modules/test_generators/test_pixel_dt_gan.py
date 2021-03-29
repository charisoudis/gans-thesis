import unittest

import torch

from modules.generators.pixel_dt_gan import PixelDTGanGenerator
from modules.partial.decoding import ExpandingBlock
from modules.partial.encoding import ContractingBlock
from utils.pytorch import enable_verbose


class TestPixelDTGan(unittest.TestCase):

    def setUp(self) -> None:
        self.c_in = 3
        self.c_out = 3
        self.c_bottleneck = 100
        self.n_contracting_blocks = 5
        self.test_gen = PixelDTGanGenerator(c_in=self.c_in, c_out=self.c_out, c_bottleneck=self.c_bottleneck,
                                            n_contracting_blocks=self.n_contracting_blocks)
        enable_verbose(self.test_gen)
        self.w_in = 256
        self.x = torch.rand(1, self.c_in, self.w_in, self.w_in)

    def test_gen_shapes(self) -> None:
        # Check output shape
        self.assertEqual((1, self.c_out, self.w_in, self.w_in), tuple(self.test_gen(self.x).shape))
        # Check bottleneck layer
        for _n, _l in self.test_gen.named_children():
            # Check if bottleneck is located in the correct index
            self.assertTrue(isinstance(_l[self.n_contracting_blocks - 1], ContractingBlock))
            self.assertTrue(isinstance(_l[self.n_contracting_blocks], ExpandingBlock))

            # Check bottleneck channels
            self.assertEqual(self.c_bottleneck, _l[self.n_contracting_blocks - 1].contracting_block[0].out_channels)
            self.assertEqual(self.c_bottleneck, _l[self.n_contracting_blocks].expanding_block[0].in_channels)

            # Check bottleneck kernel
            w_before_bottleneck = self.w_in // (2 ** (self.n_contracting_blocks - 1))
            self.assertListEqual([self.c_bottleneck, 1024, w_before_bottleneck, w_before_bottleneck],
                                 list(_l[self.n_contracting_blocks - 1].contracting_block[0].weight.shape))
            break

    def test_loss_can_run(self) -> None:
        # TODO
        pass
