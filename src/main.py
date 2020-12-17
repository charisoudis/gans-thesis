import argparse
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# noinspection PyProtectedMember
from torch.utils.data import random_split
from torchvision.transforms import transforms

from dataset.deep_fashion import ICRBCrossPoseDataset
from modules.discriminators.cycle_gan import CycleGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator
from utils.command_line_logger import CommandLineLogger


def cli_parse() -> argparse:
    """
    Parse command line arguments.
    :return: {argparse}
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--sum', action='store_true', dest='sum',
                        help='sum the integers (default: find the max)')
    parser.add_argument('--value', type=int, default=1, help='sum the integers (default: find the max)')

    return parser.parse_args()


def main():
    d = CycleGANDiscriminator(c_in=6, c_hidden=8)
    print(d)

    g = CycleGANGenerator(c_in=3, c_out=3)
    print(g)
    exit(0)

    logger = CommandLineLogger(log_level='debug')
    # logger.log_format = "> %(log_color)s%(message)s%(reset)s"
    logger.info('execution started')

    tensor_in = torch.ones(10, 1, 28, 28, device='cpu')
    tensor_out = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2, padding=1).to('cpu')(tensor_in)
    logger.info('tensor_out.shape = ' + str([*tensor_out.shape]))
    logger.info('tensor_out.max = ' + str(tensor_out.detach().cpu().max().item()))

    logger.debug(torch.cuda.get_device_capability(device=0).__str__())


# noinspection DuplicatedCode
def preview_icrb_images():
    dataset = ICRBCrossPoseDataset(image_transforms=transforms.Compose([transforms.ToTensor()]), pose=True)
    dataset_len = len(dataset)

    # Get splits
    split_lengths = [math.floor(dataset_len * 0.9), math.ceil(dataset_len * 0.1)]
    train_set, test_set = random_split(dataset, lengths=split_lengths, generator=torch.Generator().manual_seed(42))
    dataset.logger.info(f'len(train_set) = {len(train_set)} | len(test_set) = {len(test_set)}')

    # Check a pair of both
    train_pair_img_1, train_pair_img_2, train_pair_pose_2 = train_set[1234]
    test_pair_img_1, test_pair_img_2, test_pair_pose_2 = test_set[1234]
    dataset.logger.info(f'train_set[1234] = {str((train_pair_img_1.shape, train_pair_img_2.shape))}')
    dataset.logger.info(f'test_set[1234] = {str((test_pair_img_1.shape, test_pair_img_2.shape))}')

    plt.imshow(torch.cat((train_pair_img_1, train_pair_img_2), dim=2).permute(1, 2, 0))
    plt.show()
    plt.imshow(torch.cat((train_pair_img_2, train_pair_pose_2), dim=2).permute(1, 2, 0))
    plt.show()
    plt.imshow(torch.cat((test_pair_img_1, test_pair_img_2), dim=2).permute(1, 2, 0))
    plt.show()
    plt.imshow(torch.cat((test_pair_img_2, test_pair_pose_2), dim=2).permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    # args = cli_parse()
    # main()

    preview_icrb_images()
    pass
