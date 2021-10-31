import gc
import math
import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor

from modules.partial.decoding import ChannelsProjectLayer
from modules.partial.normalization import BatchStd
from utils import pytorch
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import get_total_params, get_gradient_penalty
from utils.string import to_human_readable
from utils.train import get_alpha_curve


class StyleGanDiscriminatorBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, is_initial_block: bool = False, initial_shape: int = 4):
        self.locals = locals()
        del self.locals['self']

        nn.Module.__init__(self)

        self.is_initial_block = is_initial_block
        if self.is_initial_block:
            # noinspection PyTypeChecker
            self.disc_block = nn.Sequential(
                BatchStd(),
                nn.Conv2d(c_in + 1, c_out, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(c_out, c_out, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(c_out * (initial_shape - 1) ** 2, 1)
            )
        else:
            # noinspection PyTypeChecker
            self.disc_block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc_block(x)


class StyleGanDiscriminator(nn.Module, BalancedFreezable, Verbosable):
    """
    StyleGanDiscriminator Class:
    Discriminator of StyleGAN v1 (identical to ProGAN's one).
    """

    def __init__(self, resolution: int, c_in: int = 3, c_base: int = 2048, c_max: int = 512, c_decay: float = 1.0,
                 adv_criterion: Optional[str] = None, lambda_gp: float = 10.0, alpha_multiplier: float = 10.0,
                 num_iters: int = 2, use_gradient_penalty: bool = False, logger: Optional[CommandLineLogger] = None):
        """
        StyleGanDiscriminator class constructor.
        :param (int) resolution: network's input resolution
        :param (int) c_in: number of channels expected in the input tensor
        :param (int) c_base: base number of channels (used in channels calculations)
        :param (int) c_max: max number of channels (used in channels calculations)
        :param (float) c_decay: channel decay parameter (<=1.0 - defaults to 1.0)
        :param (optional) adv_criterion: str description of desired default adversarial criterion (e.g. 'MSE', 'BCE',
                                         'BCEWithLogits', etc.). If None, then it must be set in the respective function
                                         call.
        :param (float) lambda_gp: weighting coefficient of gradient penalty when computing the loss
        :param (float) alpha_multiplier: multiplier that defines the smoothness of the alpha curve
                                         (1=linear, ..., 10=smooth, ..., 1000=delta)
        :param (int) num_iters: number of iterations
        :param (bool) use_gradient_penalty: set to True to have the discriminator compute gradient penalty alongside its
                                            loss
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        # Save arguments
        self.locals = locals()
        del self.locals['self']

        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)

        # Initialize Discriminator blocks
        for bi in range(2, int(math.log2(resolution)) + 1):
            block_index = bi - 2
            block_c_in, block_c_out = self._get_c_hidden(bi), self._get_c_hidden(bi - 1)
            setattr(self, f'fromRGB{block_index}',
                    nn.Sequential(ChannelsProjectLayer(c_in=c_in, c_out=block_c_in), nn.LeakyReLU(0.2)))
            setattr(self, f'block{block_index}',
                    StyleGanDiscriminatorBlock(block_c_in, block_c_out, is_initial_block=block_index == 0))
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        # Save args
        self.resolution = self.locals['resolution']
        self.verbose_enabled = False
        self.adv_criterion = None
        if adv_criterion is not None:
            if hasattr(nn, f'{adv_criterion}Loss'):
                self.adv_criterion = getattr(nn, f'{adv_criterion}Loss')()
            elif hasattr(pytorch, f'{adv_criterion}Loss'):
                self.adv_criterion = getattr(pytorch, f'{adv_criterion}Loss')()
            else:
                raise RuntimeError(f'adv_criterion="{adv_criterion}" could be found (tried torch.nn and utils.pytorch)')

        # Fade coefficient
        self.alpha_curve = get_alpha_curve(num_iters=num_iters, alpha_multiplier=alpha_multiplier)
        self.alpha_index = 0
        self.alpha = self.alpha_curve[self.alpha_index]

        self.logger = logger if logger is not None else \
            CommandLineLogger(log_level='debug', name=self.__class__.__name__)

    def _get_c_hidden(self, i: int):
        """
        Get no. of filters based on below formulae
        """
        return min(int(self.locals['c_base'] / (2.0 ** (i * self.locals['c_decay']))), self.locals['c_max'])

    @property
    def nparams_hr(self):
        return to_human_readable(get_total_params(self))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through StyleGAN Discriminator: given a batch of images, it computes a batch of realness
        probabilities.
        :param (torch.Tensor) x: input image of shape (N,C_in,H,W)
        :return: a torch.Tensor object with probabilities, of shape (N,1)
        """
        if self.verbose_enabled:
            self.logger.debug('_: ' + str(x.shape))

        # Get current alpha value
        alpha = self.alpha
        ################################################################################################################
        # ############################################### DEV LOGGING ##################################################
        ################################################################################################################
        # if abs(alpha - 0.5) < 1e-3:
        #     self.logger.debug(f'[DISC] alpha={alpha} (alpha_index={self.alpha_index}, len()={len(self.alpha_curve)})')
        # elif abs(alpha - 0.99) < 1e-3:
        #     self.logger.debug(f'[DISC] alpha={alpha} (alpha_index={self.alpha_index}, len()={len(self.alpha_curve)})')
        #
        # return torch.zeros(x.shape[0], 1, device=x.device, requires_grad=True)
        ################################################################################################################

        bi = int(math.log2(self.resolution))
        block_index = bi - 2
        # Edge case: alpha == 1 or one block only
        fromRGB = getattr(self, f'fromRGB{block_index}')
        block = getattr(self, f'block{block_index}')
        # self.logger.debug(f'x_new <-- block{block_index}(c_in={block.locals["c_in"]}, c_out={block.locals["c_out"]}) '
        #                   f'<-- fromRGB{block_index} <-- x: {x.shape}')
        x_new = fromRGB(x.clone())
        x_new = block(x_new)
        if block_index == 0:
            # self.logger.debug(f'x: {x_new.shape}')
            return x_new
        # Mix old and new x
        # self.logger.debug(f'x <- 1-alpha={(1-alpha)} - fromRGB{block_index - 1} <-- downsample <-- x: {x.shape}')
        # self.logger.debug(f'x <- alpha={alpha} - x_new: {x_new.shape}')
        x = self.downsample(x)
        x = getattr(self, f'fromRGB{block_index - 1}')(x)
        x = (1.0 - alpha) * x + alpha * x_new
        # Pass through the rest blocks
        for bi in reversed(range(2, int(math.log2(self.resolution)))):
            block_index = bi - 2
            block = getattr(self, f'block{block_index}')
            # self.logger.debug(f'block{block_index}(c_in={block.locals["c_in"]}, c_out={block.locals["c_out"]}) <-- '
            #                   f'x: {x.shape}')
            x = block(x)
        # self.logger.debug(f'x: {x.shape}')
        return x

    def get_loss_on_batch(self, images: Tensor, is_real: bool, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss (method compatible for usage by generator to compute its own loss).
        :param (torch.Tensor) images: a batch of real or fake images, of shape (N, C, H, W)
        :param (bool) is_real: images' source
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value
        """
        # Setup criterion
        criterion = self.adv_criterion if criterion is None else criterion
        # Proceed with loss calculation
        predictions = self(images)
        # print('DISC OUTPUT SHAPE: ' + str(predictions_on_fake.shape))
        if type(criterion) == torch.nn.modules.loss.BCELoss:
            predictions = nn.Sigmoid()(predictions)
        # Get loss' second argument
        if type(criterion) == pytorch.WassersteinLoss:
            reference = -1.0 * torch.ones_like(predictions) if is_real else 1.0 * torch.ones_like(predictions)
        else:
            reference = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
        return criterion(predictions, reference)

    def get_loss(self, real: Tensor, fake: Tensor, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss.
        :param (torch.Tensor) real: image tensor of shape (N, C, H, W) from real dataset
        :param (torch.Tensor) fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        loss_on_real = self.get_loss_on_batch(real, is_real=True, criterion=criterion)
        loss_on_fake = self.get_loss_on_batch(fake, is_real=False, criterion=criterion)
        total_loss = loss_on_real + loss_on_fake
        if not self.locals['use_gradient_penalty']:
            return total_loss
        # Calculate gradient penalty and append to loss
        gradient_penalty = get_gradient_penalty(disc=self, real=real, fake=fake)
        return total_loss + self.locals['lambda_gp'] * gradient_penalty

    def get_layer_attr_names(self) -> List[str]:
        layer_names = []
        for bi in range(2, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            layer_names.append(f'fromRGB{block_index}')
            layer_names.append(f'block{block_index}')
        layer_names.append('downsample')
        return layer_names

    def plot_alpha_curve(self):
        y = self.alpha_curve
        x = np.arange(len(y))
        plt.plot(x, y, '-.')
        plt.title(f'DISC @ {self.resolution}x{self.resolution} (num_iters={self.locals["num_iters"]})')
        plt.show()

    def grow(self, num_iters: int = 2, device: Optional[str or torch.device] = None) -> 'StyleGanDiscriminator':
        """
        Function to grow the Discriminator's resolution by a factor of 2.
        :param (int) num_iters: number of iterations the new discriminator will run
        :param (str or torch.device or None) device: new network's device
        :return: a new StyleGanDiscriminator instance
        """
        self.freeze(force=True)
        # Init new Discriminator network
        new_resolution = 2 * self.resolution
        new_locals = self.locals.copy()
        new_locals['resolution'] = new_resolution
        new_locals['num_iters'] = num_iters
        new_locals['logger'] = self.logger
        new_disc = StyleGanDiscriminator(**new_locals)
        # Transfer parameter values
        for bi in range(2, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            getattr(new_disc, f'fromRGB{block_index}') \
                .load_state_dict(getattr(self, f'fromRGB{block_index}').state_dict())
            getattr(new_disc, f'block{block_index}').load_state_dict(getattr(self, f'block{block_index}').state_dict())
        # Flush old network
        del self.downsample
        for bi in range(2, int(math.log2(self.resolution)) + 1):
            block_index = bi - 2
            delattr(self, f'fromRGB{block_index}')
            delattr(self, f'block{block_index}')
        gc.collect()
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1.0)
        # Return the initialized network
        return new_disc.to(device=device)


if __name__ == '__main__':
    # 4x4
    _resolution = 4
    _disc4 = StyleGanDiscriminator(resolution=_resolution, c_in=3, adv_criterion='Wasserstein',
                                   use_gradient_penalty=True)
    print(f'4x4 Params: {to_human_readable(get_total_params(_disc4))}')
    # enable_verbose(_disc4)
    # print(_disc4)
    time.sleep(0.5)
    _y = _disc4(torch.randn(4, 3, _disc4.resolution, _disc4.resolution))
    time.sleep(0.5)
    _disc4.logger.info(f'_y.shape={_y.shape}')

    # print(f'loss={_disc4.get_loss(real=torch.ones(4, 3, 4, 4), fake=torch.ones(4, 3, 4, 4))}')
    # exit(0)

    # 8x8
    _disc8 = _disc4.grow()
    print(f'8x8 Params: {to_human_readable(get_total_params(_disc8))}')
    # enable_verbose(_disc8)
    # print(_disc8)
    time.sleep(0.5)
    _y = _disc8(torch.randn(4, 3, _disc8.resolution, _disc8.resolution))
    time.sleep(0.5)
    _disc8.logger.info(f'_y.shape={_y.shape}')

    # 16x16
    _disc16 = _disc8.grow()
    print(f'16x16 Params: {to_human_readable(get_total_params(_disc16))}')
    # enable_verbose(_disc16)
    # print(_disc16)
    time.sleep(0.5)
    _y = _disc16(torch.randn(4, 3, _disc16.resolution, _disc16.resolution))
    time.sleep(0.5)
    _disc16.logger.info(f'_y.shape={_y.shape}')

    # 32x32
    _disc32 = _disc16.grow()
    print(f'32x32 Params: {to_human_readable(get_total_params(_disc32))}')
    # enable_verbose(_disc32)
    # print(_disc32)
    time.sleep(0.5)
    _y = _disc32(torch.randn(4, 3, _disc32.resolution, _disc32.resolution))
    time.sleep(0.5)
    _disc32.logger.info(f'_y.shape={_y.shape}')

    # 64x64
    _disc64 = _disc32.grow()
    print(f'64x64 Params: {to_human_readable(get_total_params(_disc64))}')
    # enable_verbose(_disc64)
    # print(_disc64)
    time.sleep(0.5)
    _y = _disc64(torch.randn(4, 3, _disc64.resolution, _disc64.resolution))
    time.sleep(0.5)
    _disc64.logger.info(f'_y.shape={_y.shape}')

    # 128x128
    _disc128 = _disc64.grow()
    print(f'128x128 Params: {to_human_readable(get_total_params(_disc128))}')
    # enable_verbose(_disc128)
    # print(_disc128)
    time.sleep(0.5)
    _y = _disc128(torch.randn(4, 3, _disc128.resolution, _disc128.resolution))
    time.sleep(0.5)
    _disc128.logger.info(f'_y.shape={_y.shape}')

    exit(0)

    # Compute loss
    _real = torch.randn(4, 3, 64, 64)
    _fake = torch.randn(4, 3, 64, 64)
    # print(__disc)
    _loss = _disc64.get_loss(real=_real, fake=_fake, criterion=nn.BCELoss())
    print(_loss)
