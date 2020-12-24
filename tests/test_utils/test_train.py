import os
import sys
import unittest

import torch
import torch.nn as nn
from IPython import get_ipython
from torchvision.models import inception_v3

from modules.generators.pgpg import PGPGGenerator
from utils.torch import get_total_params
from utils.train import get_adam_optimizer, load_model_chkpt


class TestTrainUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.inside_colab = 'google.colab' in sys.modules or \
                            'google.colab' in str(get_ipython()) or \
                            'COLAB_GPU' in os.environ
        self.chkpts_root: str = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints' if not self.inside_colab \
            else '/content/drive/MyDrive/Model Checkpoints'
        self.cur_step = 1024
        self.batch_size = 10
        self.conv_weight_value = 1.5

    def test_load_model_chkpt(self) -> None:
        gen = PGPGGenerator(c_in=6, c_out=3, w_in=128, h_in=128).cpu()
        gen_opt = get_adam_optimizer(gen)

        def _weights_init(_m: nn.Module) -> None:
            if isinstance(_m, nn.Conv2d) or isinstance(_m, nn.ConvTranspose2d):
                torch.nn.init.constant_(_m.weight, self.conv_weight_value)

        gen = gen.apply(_weights_init)
        torch.save({
            'gen': gen.state_dict(),
            'gen_opt': gen_opt.state_dict(),
        }, f'{self.chkpts_root}/pgpg_{self.cur_step}_{self.batch_size}.pth')

        # Signature of load_model_chkpt:  load_model_chkpt(model, model_name, dict_key, model_opt, chkpts_root)
        self.assertRaises(AssertionError, load_model_chkpt, gen, 'pgpg1', 'gen', gen_opt)
        self.assertRaises(AssertionError, load_model_chkpt, gen, 'pgpg', 'gen1', gen_opt)
        try:
            total_images_in_checkpoint = load_model_chkpt(gen, 'pgpg', 'gen', gen_opt)
        except AssertionError:
            self.fail("load_model_chkpt() raised AssertionError")
        self.assertEqual(self.cur_step * self.batch_size, total_images_in_checkpoint)

        # Check loading of state dict to a new model
        gen2 = PGPGGenerator(c_in=6, c_out=3, w_in=128, h_in=128).cpu()
        load_model_chkpt(gen2, 'pgpg', 'gen')

        def _check_weights(_m):
            if isinstance(_m, nn.Conv2d) or isinstance(_m, nn.ConvTranspose2d):
                self.assertTrue(torch.all(_m.weight.data.eq(self.conv_weight_value * torch.ones_like(_m.weight.data))))

        gen2.apply(_check_weights)

        # Try loading inception checkpoint
        inception = inception_v3(pretrained=False, init_weights=False)
        try:
            load_model_chkpt(inception, 'inception_v3')
        except AssertionError:
            self.fail("load_model_chkpt(inception, 'inception_v3') raised AssertionError")

    def test_get_adam_optimizer(self) -> None:
        # Training of a single model
        model = nn.Conv2d(3, 10, 3, bias=False)
        optimizer = get_adam_optimizer(model)
        self.assertEqual(type(optimizer), torch.optim.Adam)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params']), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params'][0].view(-1)), get_total_params(model))

        # Joint training of two models
        model1 = nn.Conv2d(3, 10, 3, bias=False)
        model2 = nn.Conv2d(3, 5, 2, bias=False)
        optimizer = get_adam_optimizer(model1, model2, lr=1e-2)
        self.assertEqual(torch.optim.Adam, type(optimizer))
        self.assertEqual(1, len(optimizer.param_groups))
        self.assertEqual(2, len(optimizer.param_groups[0]['params']))
        self.assertEqual(get_total_params(model1), len(optimizer.param_groups[0]['params'][0].view(-1)))
        self.assertEqual(get_total_params(model2), len(optimizer.param_groups[0]['params'][1].view(-1)))
        self.assertEqual(1e-2, optimizer.param_groups[0]['lr'])
        model1_wb = nn.Conv2d(3, 10, 3)
        model2_wb = nn.Conv2d(3, 5, 2)
        optimizer = get_adam_optimizer(model1_wb, model2_wb, lr=1e-5)
        self.assertEqual(1, len(optimizer.param_groups))
        self.assertEqual(4, len(optimizer.param_groups[0]['params']))
        self.assertEqual(get_total_params(model1), len(optimizer.param_groups[0]['params'][0].view(-1)))
        self.assertEqual(10, len(optimizer.param_groups[0]['params'][1].view(-1)))  # 1st model's bias params
        self.assertEqual(get_total_params(model2), len(optimizer.param_groups[0]['params'][2].view(-1)))
        self.assertEqual(5, len(optimizer.param_groups[0]['params'][3].view(-1)))  # 2nd model's bias params

    def tearDown(self) -> None:
        chkpt_path = f'{self.chkpts_root}/pgpg_{self.cur_step}_{self.batch_size}.pth'
        if os.path.exists(chkpt_path):
            os.remove(chkpt_path)
