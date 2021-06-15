import abc
import json
import os
from abc import ABCMeta
from typing import Optional, Union, Dict, List, Sequence

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL.Image import Image
from scipy.interpolate import make_interp_spline
from torch import Tensor

from utils.command_line_logger import CommandLineLogger
from utils.filesystems.gdrive import GDriveModel
from utils.ifaces import FilesystemModel, Configurable, Evaluable, Visualizable, FilesystemFolder, FilesystemFile
from utils.plot import pltfig_to_pil
from utils.pytorch import get_total_params
from utils.string import to_human_readable


class IModule(FilesystemModel, Configurable, Evaluable, Visualizable, metaclass=abc.ABCMeta):
    """
    IModule Class/Interface:
    Default module interface, implementing FilesystemModel for cloud/local checkpoints syncing, Configurable to
    support configurations, Evaluable for running GAN evaluation metrics, Visualizable for creating & storing progress
    as well as model outputs.
    """

    DefaultConfiguration = None

    def __init__(self, config_id: Optional[str] = None, log_level: Optional[str] = None,
                 evaluator: Optional = None, reproducible_indices: Sequence = (0, -1)):
        """
        IGanModule class constructor.
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (str) log_level: CommandLineLogger's log level
        :param (optional) evaluator: GANEvaluator instance or similar
        :param (Sequence) reproducible_indices: indices tuple or list, defaults to (0, -1)
        """
        # Initialize Visualizable
        Visualizable.__init__(self, indices=reproducible_indices)

        # Initialize logger
        if log_level is not None:
            self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__)

        # Load model configuration from local/cloud filesystem or use default
        self.config_id = config_id
        if config_id is None or config_id == 'default':
            configuration = self.DefaultConfiguration
        else:
            config_filepath = self.fetch_configuration(config_id=config_id)
            with open(config_filepath) as yaml_fp:
                configuration = yaml.load(yaml_fp, Loader=yaml.FullLoader)
                self.logger.debug(f'Loaded configuration with id: "{config_id}"')
        assert configuration is not None and isinstance(configuration, dict), 'Configuration has not been initialized'

        # Check evaluator
        self.evaluator = evaluator

        # Initialize generator and transforms instances
        self.gen = None
        #   - save generator transforms for visualizer
        self.gen_transforms = None
        if evaluator is not None and hasattr(evaluator, 'gen_transforms'):
            self.gen_transforms = evaluator.gen_transforms

        # Save configuration in instance
        self._configuration = configuration
        self._nparams = None
        self.gen_losses = []
        self.disc_losses = []

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
    # --------------
    # Configurable
    # -------------
    #

    def configuration(self) -> dict:
        return {**self._configuration, 'nparams': self.nparams, 'nparams_hr': self.nparams_hr}

    def load_configuration(self, configuration: dict) -> None:
        raise NotImplementedError

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

    def visualize(self, reproducible: bool = False) -> Image:
        raise NotImplementedError

    def visualize_indices(self, indices: int or Sequence) -> Image:
        raise NotImplementedError

    def visualize_losses(self, dict_keys: tuple, upload: bool = False, preview: bool = False) -> List[Image]:
        """
        Fetch the losses from every checkpoint and plot them nicely.
        :param (tuple) dict_keys: e.g. ('gen_loss', 'disc_loss') --> 2 images, each with the respective loss
                                       (('gen_loss', 'disc_loss')) --> 1 image with both losses
                                       (('g1_loss', 'g2_loss'), 'disc_loss') --> 2 images, the 1st with 2 losses
        :param (bool) upload: set to True to have all the resulting images been uploaded to GoogleDrive
        :param (bool) preview: set to True to plot images using matplotlib
        :return: a list of PIL.Image objects
        """
        assert isinstance(self, FilesystemModel), 'Model must implement utils.ifaces.FilesystemFolder to visualize' + \
                                                  'checkpoint losses and upload produced images to cloud'
        # Download all checkpoints locally
        chkpt_files = {}
        chkpts_dict = self.list_all_checkpoints()
        for epoch, epoch_chkpts in chkpts_dict.items():
            # Check if file is in local filesystem and is a chkpt file
            _files = []
            for _f in epoch_chkpts:
                if isinstance(_f, FilesystemFile) and _f.name.endswith('.pth'):
                    if not _f.is_downloaded:  # and not _f.download(in_parallel=False, show_progress=True):
                        pass
                        # raise FileNotFoundError('File not found in GoogleDrive or FAILed to download')
                    _files.append(_f)
            # Check if epoch yielded on checkpoints
            if len(_files) == 0:
                self.logger.warning(f'Epoch #{epoch} has no checkpoint (.pth) files')
                continue
            # Append file to outer list
            chkpt_files[epoch] = _files.copy()

        # Initialize dicts
        losses_dict = {}
        for ki, key_or_keys in enumerate(dict_keys):
            _keys = (key_or_keys,) if type(key_or_keys) == str else key_or_keys
            # Create list
            if f'img_{ki}' not in losses_dict.keys():
                losses_dict[f'img_{ki}'] = {_k: {} for _k in _keys}

        # Process each chkpt file and extract the required keys
        for epoch, epoch_chkpts in chkpt_files.items():
            epoch_chkpt: FilesystemFile
            for epoch_chkpt in epoch_chkpts:
                # Load chkpt file
                chkpt_dict = torch.load(epoch_chkpt.path, map_location='cpu')
                # Process it and append to images data
                for ki, key_or_keys in enumerate(dict_keys):
                    _keys = (key_or_keys,) if type(key_or_keys) == str else key_or_keys
                    _ii = f'img_{ki}'  # image index
                    # Fetch values for corresponding keys
                    for _k in _keys:
                        #   - initialize epoch dict
                        if epoch not in losses_dict[_ii][_k].keys():
                            losses_dict[_ii][_k][epoch] = []
                        #   - check if key exists
                        if _k not in chkpt_dict.keys():
                            losses_dict[_ii][_k][epoch].append(np.NaN)
                        else:
                            losses_dict[_ii][_k][epoch].append(chkpt_dict[_k])

        # Set matplotlib params
        matplotlib.rcParams["font.family"] = 'JetBrains Mono'
        # Configure matplotlib for pretty plots
        plt.rcParams["font.family"] = 'JetBrains Mono'

        # Create a subfolder inside "Visualizations" folder to store aggregated metrics plots
        vis_losses_folder = self.visualizations_fs_folder.subfolder_by_name_or_create(
            folder_name='Losses Plots')
        filename_suffix = f'epochs={tuple(chkpts_dict.keys())[0]}_{tuple(chkpts_dict.keys())[-1]}.jpg'

        # Save loss plots as images & display inline
        _returns = []
        for ki, key_or_keys in enumerate(dict_keys):
            _keys = (key_or_keys,) if type(key_or_keys) == str else key_or_keys
            _ii = f'img_{ki}'  # image index
            _idata = losses_dict[_ii]  # image data (e.g. {'gen_loss': {0: [0.1, 0.4, ...], 1: [0.2, ...], ...},
            #                   'disc_loss: {....}

            # Create a new figure
            plt.figure(figsize=(10, 5), dpi=300, clear=True)

            # Plot each curve
            curve_key: str
            curve_name: dict
            for curve_name, curve_data in _idata.items():
                print(f'plotting {curve_name} (len = {len(curve_data)})')

                # x-axis | y-axis
                curve_x = []
                curve_y = []
                for epoch, epoch_values in curve_data:
                    for vi, v in enumerate(epoch_values):
                        curve_x.append(epoch + vi / len(epoch_values))
                        curve_y.append(v)

                # Plot curve (smooth line + actual points)
                x_new = np.linspace(curve_x[0], curve_x[-1], 300)
                plt.plot(x_new, make_interp_spline(curve_x, curve_y, k=3)(x_new), '-.', color='#2a9ceb')
                plt.plot(curve_x, curve_y, 'o', color='#1f77b4')

            # Set figure title
            plt_title = f'{" vs. ".join(_keys)}'
            plt_subtitle = f'{filename_suffix.replace("_", " to ").replace("=", ": ").replace(".jpg", "")}'
            plt.suptitle(f'{plt_title}', y=0.97, fontsize=12, fontweight='bold')
            plt.title(f'{plt_subtitle}', pad=10., fontsize=10, )

            # Get PIL image
            pil_img = pltfig_to_pil(plt.gcf())
            _returns.append(pil_img)
            # Save visualization file
            if upload:
                filepath = '{}/{}_{}'.format(vis_losses_folder.local_root, f'{"__vs__".join(_keys)}'.strip().lower(),
                                             filename_suffix)
                is_update = os.path.exists(filepath)
                pil_img.save(filepath)
                # Upload to Google Drive
                vis_losses_folder.upload_file(local_filename=filepath, in_parallel=False, is_update=is_update)
                self.logger.debug(f'Loss image from {filepath} uploaded successfully!')
            if preview:
                plt.show()
        return _returns

    def visualize_metrics(self, upload: bool = False, preview: bool = False) -> List[Image]:
        assert isinstance(self, FilesystemModel), 'Model must implement utils.ifaces.FilesystemFolder to visualize' + \
                                                  'metrics and upload produced images to cloud'
        # Init metric lists
        x, fids, iss, f1s, ssims = ([], [], [], [], [])
        epoch_metrics_dict = self.list_all_metrics()
        for epoch, epoch_metrics in epoch_metrics_dict.items():
            # Check if file is in local filesystem and is a metric file
            _files = []
            for _f in epoch_metrics:
                if isinstance(_f, FilesystemFile) and _f.name.endswith('.json'):
                    if not _f.is_downloaded and not _f.download(in_parallel=False, show_progress=False):
                        raise FileNotFoundError('File not found in GoogleDrive or FAILed to download')
                    _files.append(_f)

            if len(_files) == 0:
                self.logger.warning(f'Epoch #{epoch} has no metrics JSON files')
                continue

            # We do not know beforehand the number of metric files inside each epoch
            x_step = 1 / len(_files)
            x_init = epoch

            metric_file: FilesystemFile
            for _fi, metric_file in enumerate(_files):
                # Fetch metrics from file
                self.logger.debug(f'opening metrics file: {metric_file.path}')
                with open(metric_file.path) as json_fp:
                    metrics = json.load(json_fp)
                # Update lists
                fids.append(metrics['fid'])
                iss.append(metrics['is'])
                f1s.append(metrics['f1'] if not np.isnan(metrics['f1']) else 0)
                ssims.append(metrics['ssim'])
                # Set correct x value
                x.append(x_init + _fi * x_step)
        if len(epoch_metrics_dict.keys()) == 0:
            self.logger.debug('No metrics found. Returning...')
            return []

        # Set matplotlib params
        matplotlib.rcParams["font.family"] = 'JetBrains Mono'
        # Configure matplotlib for pretty plots
        plt.rcParams["font.family"] = 'JetBrains Mono'

        # Create a subfolder inside "Visualizations" folder to store aggregated metrics plots
        vis_metrics_folder = self.visualizations_fs_folder.subfolder_by_name_or_create(folder_name='Metrics Plots')
        filename_suffix = f'epochs={tuple(epoch_metrics_dict.keys())[0]}_{tuple(epoch_metrics_dict.keys())[-1]}.jpg'

        # Save metric plots as images & display inline
        _returns = []
        for metric_name, metric_data in zip(('  FID', '  IS', '  F1', '  SSIM'), (fids, iss, f1s, ssims)):
            # Create a new figure
            plt.figure(figsize=(10, 5), dpi=300, clear=True)
            # Set data
            # plt.plot(x, metric_data)
            x_new = np.linspace(x[0], x[-1], 300)
            plt.plot(x_new, make_interp_spline(x, metric_data, k=3)(x_new), '-.', color='#2a9ceb')
            plt.plot(x, metric_data, 'o', color='#1f77b4')
            # Set title
            plt_title = f'{metric_name} Metric'
            plt_subtitle = f'{filename_suffix.replace("_", " to ").replace("=", ": ").replace(".jpg", "")}'
            plt.suptitle(f'{plt_title}', y=0.97, fontsize=12, fontweight='bold')
            plt.title(f'{plt_subtitle}', pad=10., fontsize=10, )
            # Get PIL image
            pil_img = pltfig_to_pil(plt.gcf())
            _returns.append(pil_img)
            # Save visualization file
            if upload:
                filepath = '{}/{}_{}'.format(vis_metrics_folder.local_root, metric_name.strip().lower(),
                                             filename_suffix)
                is_update = os.path.exists(filepath)
                pil_img.save(filepath)
                # Upload to Google Drive
                vis_metrics_folder.upload_file(local_filename=filepath, in_parallel=False, is_update=is_update)
                self.logger.debug(f'Metric file from {filepath} uploaded successfully!')
            if preview:
                plt.show()
        return _returns


class IGModule(GDriveModel, IModule, metaclass=ABCMeta):
    """
    IGModule Class/Interface:
    Default Google Drive module interface, implementing GDriveModel for cloud checkpoints/confs/metrics, and IModule for
    the rest of module abilities.
    """

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 log_level: str = 'info', dataset_len: Optional[int] = None, evaluator: Optional = None,
                 reproducible_indices: Sequence = (0, -1)):
        """
        IGModule class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.gdrive.GDriveFolder` object to download /
                                                           upload model checkpoints and metrics from / to local or
                                                           remote (Google Drive) filesystem
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (str) log_level: CommandLineLogger's log level
        :param (optional) evaluator:
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param (Sequence) reproducible_indices: indices tuple or list, defaults to (0, -1)
        """
        # Initialize logger
        self.logger = CommandLineLogger(log_level=log_level, name=self.__class__.__name__)

        # Instantiate GDriveModel
        model_name_from_class = self.__class__.__name__.lower()
        model_name = os.environ.get(f'NEW_MODEL_NAME__{model_name_from_class}', model_name_from_class)
        if model_name != model_name_from_class:
            self.logger.info(f'Using "{model_name}" instead of "{model_name_from_class}" as model\'s name')
        model_fs_folder = model_fs_folder_or_root if model_fs_folder_or_root.name.endswith(model_name) else \
            model_fs_folder_or_root.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=True)
        if model_fs_folder is None:
            self.logger.warning('Filesystem folder for model not found. Creating one now.')
            model_fs_folder = model_fs_folder_or_root.create_subfolder(f'model_name={model_name}',
                                                                       force_create_local=True)
            assert model_fs_folder is not None
        GDriveModel.__init__(self, model_fs_folder=model_fs_folder, logger=self.logger,
                             model_name=model_name, dataset_len=dataset_len)

        # Initialize IModule
        IModule.__init__(self, config_id=config_id, evaluator=evaluator, reproducible_indices=reproducible_indices)


class IGanGModule(IGModule, metaclass=abc.ABCMeta):
    """
    IGanGModule Class/Interface.
    Default Google Drive module interface, implementing GDriveModel for cloud checkpoints/confs/metrics, and IModule for
    the rest of module abilities.
    """

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 device: Optional[Union[torch.device, str]] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, -1),
                 evaluator: Optional = None, **evaluator_kwargs):
        """
        IGanModule class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.gdrive.GDriveFolder` object to download /
                                                           upload model checkpoints and metrics from / to local or
                                                           remote (Google Drive) filesystem
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (str) device: the device used for training (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (str) log_level: CommandLineLogger's log level
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param (Sequence) reproducible_indices: indices tuple or list, defaults to (0, -1)
        :param (optional) evaluator: utils.metrics.GanEvaluator instance
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        # Check evaluator
        if evaluator is None and device is not None:
            try:
                from utils.metrics import GanEvaluator
                self.evaluator = GanEvaluator(model_fs_folder_or_root=model_fs_folder_or_root, device=device,
                                              **evaluator_kwargs)
            except TypeError or AttributeError:
                self.evaluator = None
        else:
            self.evaluator = evaluator
        if evaluator is not None and dataset_len is None:
            self.dataset_len = len(evaluator.dataset)

        # Initialize parent
        IGModule.__init__(self, model_fs_folder_or_root=model_fs_folder_or_root, config_id=config_id,
                          log_level=log_level, dataset_len=dataset_len, evaluator=evaluator,
                          reproducible_indices=reproducible_indices)

    def load_configuration(self, configuration: dict) -> None:
        raise NotImplementedError("Found no practical way to change configuration in an online manner")
