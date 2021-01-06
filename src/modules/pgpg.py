from typing import Optional, Tuple, Union, Dict

import torch
import yaml
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.deep_fashion import ICRBDataset, ICRBCrossPoseDataset
from modules.discriminators.patch_gan import PatchGANDiscriminator
from modules.generators.pgpg import PGPGGenerator
from utils.data import plt_to_pil
from utils.filesystems.gdrive import GDriveModel
from utils.filesystems.local import LocalFilesystem, LocalCapsule, LocalFolder
from utils.ifaces import Configurable, Evaluable, FilesystemFolder
from utils.metrics import GanEvaluator
from utils.pytorch import get_total_params, invert_transforms
from utils.string import to_human_readable
from utils.train import weights_init_naive, get_adam_optimizer


class PGPG(nn.Module, GDriveModel, Configurable, Evaluable):
    """
    PGPG Class:
    This class is used to access and use the entire PGPG model (implemented according to the paper "Pose-Guided Person
    Image Generation" as a `nn.Module` instance but with the additional functionality provided from inheriting
    `utils.gdrive.GDriveModel`. Inheriting GDriveModel we can easily download / upload model checkpoints to
    Google Drive using GoogleDrive API's python client.
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'shapes': {
            'c_in': 3,
            'c_out': 3,
            'w_in': 128,
            'h_in': 128,
        },
        'gen': {
            'g1': {
                'c_hidden': 32,
                'n_contracting_blocks': 6,
                'c_bottleneck_down': 256,
                'use_out_tanh': True,
            },
            'g2': {
                'c_hidden': 32,
                'n_contracting_blocks': 5,
                'use_out_tanh': True,
                'use_dropout': True,
            },
            'recon_criterion': 'L1',
            'adv_criterion': 'MSE',
        },
        'disc': {
            'c_hidden': 8,
            'n_contracting_blocks': 5,
            'use_spectral_norm': True,
            'adv_criterion': 'MSE',
        },
        'opt': {
            'lr': 1e-4,
            'schedule': None
        }
    }

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 chkpt_step: Optional[int or str] = None, chkpt_batch_size: Optional[int] = None,
                 device: str = 'cpu', gen_transforms: Optional[Compose] = None,
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
        """
        PGPG class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.gdrive.GDriveFolder` object to download /
                                                           upload model checkpoints and metrics from / to local or
                                                           remote (Google Drive) filesystem
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (int or str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` will
                                                be loaded via `nn.Module().load_state_dict()`
        :param (int or None) chkpt_batch_size: if not `None` then the model checkpoint at the given :attr:`step` and at
                                               the given :attr:`batch_size` will be loaded via
                                               `nn.Module().load_state_dict()` call
        :param (str) device: the device used for training (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (Compose) gen_transforms: the image transforms of the dataset the generator is trained on (used in
                                         visualization)
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        # Instantiate GDriveModel class
        model_name = self.__class__.__name__.lower()
        model_fs_folder = model_fs_folder_or_root if model_fs_folder_or_root.name.endswith(model_name) else \
            model_fs_folder_or_root.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=True)
        GDriveModel.__init__(self, model_fs_folder=model_fs_folder, model_name=model_name)
        # Instantiate InceptionV3 model
        nn.Module.__init__(self)
        # Load model configuration from Google Drive or use default
        if config_id:
            config_filepath = self.fetch_configuration(config_id=config_id)
            with open(config_filepath) as yaml_fp:
                configuration = yaml.load(yaml_fp, Loader=yaml.FullLoader)
            self.config_id = config_id
        else:
            configuration = self.DefaultConfiguration
            self.config_id = None

        # Define PGPG model
        # This setup leads to 237M (G1 has ~ 120M, G2 has ~117M) learnable parameters for the entire Generator network
        shapes_conf = configuration['shapes']
        self.gen = PGPGGenerator(c_in=2 * shapes_conf['c_in'], c_out=shapes_conf['c_out'], w_in=shapes_conf['w_in'],
                                 h_in=shapes_conf['h_in'], configuration=configuration['gen'])
        # This setup leads to 396K learnable parameters for the Discriminator network
        # NOTE: for 5 contracting blocks, output is 4x4
        disc_conf = configuration['disc']
        self.disc = PatchGANDiscriminator(c_in=2 * shapes_conf['c_in'],
                                          n_contracting_blocks=disc_conf['n_contracting_blocks'],
                                          use_spectral_norm=bool(disc_conf['use_spectral_norm']))
        self.disc_adv_criterion = getattr(nn, f'{disc_conf["adv_criterion"]}Loss')()
        # Move models to GPU
        self.gen.to(device)
        self.disc.to(device)
        self.device = device
        # Define optimizers
        # Note: Both generators, G1 & G2, are trained using a joint optimizer
        opt_conf = configuration['opt']
        self.gen_opt = get_adam_optimizer(self.gen, lr=opt_conf['lr'], betas=(0.9, 0.999))
        self.disc_opt = get_adam_optimizer(self.disc, lr=opt_conf['lr'], betas=(0.9, 0.999))
        # Define LR schedulers
        # TODO

        # Load checkpoint from Google Drive
        if chkpt_step:
            chkpt_filepath = self.fetch_checkpoint(step=chkpt_step, batch_size=chkpt_batch_size)
            self.load_state_dict(torch.load(chkpt_filepath, map_location='cpu'))
        else:
            # Initialize weights with small values
            self.gen = self.gen.apply(weights_init_naive)
            self.disc = self.disc.apply(weights_init_naive)
        # Save configuration in instance
        self._configuration = configuration
        self._nparams = None

        # Check evaluator
        if not evaluator:
            try:
                self.evaluator = GanEvaluator(model_fs_folder_or_root=model_fs_folder_or_root, device=device,
                                              **evaluator_kwargs)
            except TypeError or AttributeError:
                self.evaluator = None
        else:
            self.evaluator = evaluator

        # Save transforms for visualizer
        self.gen_transforms = evaluator.gen_transforms if evaluator else gen_transforms
        self.g1_out = None
        self.g_out = None
        self.image_1 = None
        self.image_2 = None
        self.pose_2 = None

    @property
    def nparams(self) -> int or str:
        """
        Get the total numbers of parameters of this model.
        :return: an `int` object
        """
        if not self._nparams:
            self._nparams = get_total_params(self)
        return self._nparams

    @property
    def nparams_hr(self) -> str:
        """
        Get the total numbers of parameters of this model as a human-readable string.
        :return: a `str` object
        """
        return to_human_readable(self.nparams)

    #
    # ------------
    # nn.Module
    # -----------
    #

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_opt.load_state_dict(state_dict['gen_opt'])
        self.disc.load_state_dict(state_dict['disc'])
        self.disc_opt.load_state_dict(state_dict['disc_opt'])

    def state_dict(self, *args, **kwargs) -> dict:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        return {
            'gen': self.gen.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'disc': self.disc.state_dict(),
            'disc_opt': self.disc_opt.state_dict(),
            'nparams': self.nparams,
            'nparams_hr': self.nparams_hr,
        }

    def forward(self, image_1: Tensor, image_2: Tensor, pose_2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param (Tensor) image_1: the batch of input images as a `torch.Tensor` object
        :param (Tensor) image_2: the batch of target images as a `torch.Tensor` object
        :param (Tensor) pose_2: the batch of target pose images as a `torch.Tensor` object
        :return: a 4-tuple of `torch.Tensor` objects containing (disc_loss, gen_loss, g1_out, g_out)

        -----------
        FROM COLAB:
        ----------

        # Perform forward pass from generator adn discriminator
        disc_loss, gen_loss, g1_out, g_out = pgpg(image_1, image_2, pose_2)


        """
        # Update gdrive model state
        self.gforward(image_1.shape[0])

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        self.disc_opt.zero_grad()  # Zero out gradient before backprop
        with torch.no_grad():
            _, g_out = self.gen(image_1, pose_2)
        disc_loss = self.disc.get_loss(real=image_2, fake=g_out, condition=image_1, criterion=self.disc_adv_criterion)
        disc_loss.backward(retain_graph=True)  # Update discriminator gradients
        self.disc_opt.step()  # Update discriminator weights
        # Update LR (if needed)
        # if disc_lr_scheduler_type and disc_opt_lr_scheduler:
        #     disc_opt_lr_scheduler.step(metrics=disc_loss) if disc_lr_scheduler_type == 'on_plateau' \
        #         else disc_opt_lr_scheduler.step()

        ##########################################
        ########     Update Generator     ########
        ##########################################
        self.gen_opt.zero_grad()
        g1_loss, g2_loss, g1_out, g_out = self.gen.get_loss(x=image_1, y=image_2, y_pose=pose_2.clone(), disc=self.disc,
                                                            adv_criterion=None, recon_criterion=None)
        gen_loss = g1_loss + g2_loss
        gen_loss.backward()  # Update generator gradients

        self.gen_opt.step()  # Update generator optimizer
        # Update LR (if needed)
        # if gen_lr_scheduler_type and gen_opt_lr_scheduler:
        #     gen_opt_lr_scheduler.step(metrics=gen_loss) if gen_lr_scheduler_type == 'on_plateau' \
        #         else gen_opt_lr_scheduler.step()

        # Save for visualization
        self.g1_out = g1_out[::len(g1_out) - 1].detach().cpu()
        self.g_out = g_out[::len(g_out) - 1].detach().cpu()
        self.image_1 = image_1[::len(image_1) - 1].detach().cpu()
        self.image_2 = image_2[::len(image_2) - 1].detach().cpu()
        self.pose_2 = pose_2[::len(pose_2) - 1].detach().cpu()

        return disc_loss, gen_loss, g1_out, g_out

    #
    # --------------
    # Configurable
    # -------------
    #

    def configuration(self) -> dict:
        return {**self._configuration, 'nparams': self.nparams, 'nparams_hr': self.nparams_hr}

    def load_configuration(self, configuration: dict) -> None:
        raise NotImplementedError("Found no practical way to change configuration in an online manner")

    #
    # -----------
    # Evaluable
    # ----------
    #

    def evaluate(self, metric_name: Optional[str] = None, show_progress: bool = True) \
            -> Union[Dict[str, Tensor or float], Tensor or float]:
        if not self.evaluator:
            raise AttributeError('cannot evaluate model when evaluator not set')
        return self.evaluator.evaluate(gen=self.gen, metric_name=metric_name, show_progress=show_progress)

    #
    # --------------
    # Visualizable
    # -------------
    #

    def visualize(self) -> Union[None, Image]:
        # Inverse generator transforms
        gen_transforms_inv = invert_transforms(self.gen_transforms)

        g1_out_first = gen_transforms_inv(self.g1_out[0]).float()
        g1_out_last = gen_transforms_inv(self.g1_out[-1]).float()
        g_out_first = gen_transforms_inv(self.g_out[0]).float()
        g_out_last = gen_transforms_inv(self.g_out[-1]).float()
        g2_out_fist = g_out_first - g1_out_first
        g2_out_last = g_out_last - g1_out_last

        image_1_first = gen_transforms_inv(self.image_1[0])
        pose_2_first = self.pose_2[0]  # No normalization since skip_pose_norm = True
        image_2_first = gen_transforms_inv(self.image_2[0])
        image_1_last = gen_transforms_inv(self.image_1[-1])
        pose_2_last = self.pose_2[-1]  # No normalization since skip_pose_norm = True
        image_2_last = gen_transforms_inv(self.image_2[-1])

        border = 2
        black = 0.4
        cat_images_1 = torch.cat((
            black * torch.ones(3, image_1_first.shape[1], border).float(),
            image_1_first, pose_2_first, image_2_first, g1_out_first, g2_out_fist, g_out_first,
            black * torch.ones(3, image_1_first.shape[1], border).float()
        ), dim=2).cpu()
        cat_images_2 = torch.cat((
            black * torch.ones(3, image_1_first.shape[1], border).float(),
            image_1_last, pose_2_last, image_2_last, g1_out_last, g2_out_last, g_out_last,
            black * torch.ones(3, image_1_first.shape[1], border).float()
        ), dim=2).cpu()

        cat_images = torch.cat((
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            cat_images_1,
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            1.0 * torch.ones(3, 4*border, cat_images_1.shape[2]).float(),
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
            cat_images_2,
            black * torch.ones(3, border, cat_images_1.shape[2]).float(),
        ), dim=1)

        # img = tensor_to_image(cat_images)
        # return img
        plt.rcParams["figure.figsize"] = (5, 2)
        plt.axis('off')
        plt.imshow(cat_images.permute(1, 2, 0))
        img, buf = plt_to_pil(plt.gcf())
        img.show()
        buf.close()
        # plt.show()

        # target_channels = image_1_first.shape[0]
        # if target_channels == 1:
        #     cat_images = cat_images.view([cat_images.shape[1], cat_images.shape[2]])
        # else:
        #     cat_images = cat_images.permute(1, 2, 0)
        # img = plt.imshow(cat_images, cmap='gray' if target_channels == 1 else None)

        # cat_images = torch.cat((image_1_last, pose_2_last, image_2_last, g1_out_last, g2_out_last, g_out_last),
        #                        dim=2).cpu()
        # if target_channels == 1:
        #     cat_images = cat_images.view([cat_images.shape[1], cat_images.shape[2]])
        # else:
        #     cat_images = cat_images.permute(1, 2, 0)
        # plt.imshow(cat_images, cmap='gray' if target_channels == 1 else None)
        # plt.show()


if __name__ == '__main__':
    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'

    # Via GoogleDrive API
    # _capsule = GDriveCapsule(local_gdrive_root=_local_gdrive_root, use_http_cache=True, update_credentials=True)
    # _fs = GDriveFilesystem(gcapsule=_capsule)
    # _groot = GDriveFolder.root(capsule_or_fs=_fs, update_cache=False)

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _fs = LocalFilesystem(LocalCapsule(_local_gdrive_root))
    _groot = LocalFolder.root(capsule_or_fs=_fs)

    # Define folder roots
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    # Initialize model evaluator
    _gen_transforms = ICRBDataset.get_image_transforms(target_shape=128, target_channels=3)
    _dataset = ICRBCrossPoseDataset(dataset_fs_folder_or_root=_datasets_groot, image_transforms=_gen_transforms,
                                    pose=True)
    _dl = DataLoader(dataset=_dataset, batch_size=2)
    _evaluator = GanEvaluator(model_fs_folder_or_root=_models_groot, gen_dataset=_dataset, target_index=1,
                              condition_indices=(0, 2), n_samples=2, batch_size=1,
                              f1_k=1)

    # Initialize model
    _pgpg = PGPG(model_fs_folder_or_root=_models_groot, config_id='g2_with_tanh_and_dropout',
                 # chkpt_step=1, chkpt_batch_size=1,
                 evaluator=_evaluator, device='cpu')
    # print(json.dumps(_pgpg.list_all_configurations(only_keys=('title',)), indent=4))

    _device = _pgpg.device
    # _x = torch.randn(2, 3, 128, 128).to(_device)
    _x, _y, _y_pose = next(iter(_dl))
    _disc_loss, _gen_loss, _g1_out, _g_out = _pgpg(_x, _y, _y_pose)
    print(_disc_loss.shape, _gen_loss.shape, _g1_out.shape, _g_out.shape)

    _pgpg.visualize()

    # import time
    #
    # print('starting capturing...')
    # _async_results = _pgpg.gcapture(in_parallel=True, show_progress=True)
    # for i in range(20):
    #     ready = all(_r.ready() for _r in _async_results)
    #     if not ready:
    #         print('Not ready: sleeping...')
    #         time.sleep(1)
    #     else:
    #         break
    # _uploaded_gfiles = [_r.get() for _r in _async_results]
    # print(json.dumps(_uploaded_gfiles, indent=4))
