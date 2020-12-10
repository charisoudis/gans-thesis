import argparse
import torch
import torch.nn as nn
from modules.discriminators.cycle_gan import CycleGANDiscriminator
from modules.generators.cycle_gan import CycleGANGenerator
from modules.cycle_gan import CycleGAN
from utils.command_line_logger import CommandLineLogger
from utils.pytorch import get_total_params


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


if __name__ == '__main__':
    # args = cli_parse()
    # main()

    cg = CycleGAN()
    get_total_params(cg, print_table=True, sort_desc=True)
