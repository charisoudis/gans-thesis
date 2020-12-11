import unittest
import torch
import torch.nn as nn

from utils.pytorch import get_total_params
from utils.train import get_adam_optimizer


class TestTrainUtils(unittest.TestCase):

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
        self.assertEqual(5, len(optimizer.param_groups[0]['params'][3].view(-1)))   # 2nd model's bias params
