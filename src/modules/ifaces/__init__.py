import abc
import gc
import json
import os
import time
from abc import ABCMeta
from typing import Optional, Union, Dict, List, Sequence, Tuple

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
from utils.data import unnanify
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
        self.gen_losses_permanent = []
        self.gen_losses_indices = []
        self.disc_losses = []
        self.disc_losses_permanent = []
        self.disc_losses_indices = []

    @property
    def nparams(self) -> int or str:
        """
        Get the total numbers of parameters of this model.
        :return: an `int` object
        """
        if not self._nparams:
            # noinspection PyTypeChecker
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

    def visualize_losses(self, dict_keys: tuple, upload: bool = False, preview: bool = False,
                         colors: Optional[List[Tuple[str, str]]] = None, extract_dicts: bool = False) -> List[Image]:
        """
        Fetch the losses from every checkpoint and plot them nicely.
        :param (tuple) dict_keys: e.g. ('gen_loss', 'disc_loss') --> 2 images, each with the respective loss
                                       (('gen_loss', 'disc_loss')) --> 1 image with both losses
                                       (('g1_loss', 'g2_loss'), 'disc_loss') --> 2 images, the 1st with 2 losses
        :param (bool) upload: set to True to have all the resulting images been uploaded to GoogleDrive
        :param (bool) preview: set to True to plot images using matplotlib
        :param (optional) colors: a list of tuples wherein the first is the color of the line and the second is the
                                  color of the point (default to None == [(blue1, blue2), (orange1, orange2),
                                                                          (green1, green2), (purple1, purple2), (greys)]
        :param (bool) extract_dicts: set to True to save stripped versions of model checkpoints in separate files for
                                     reusability
        :return: a list of PIL.Image objects
        """
        assert isinstance(self, FilesystemModel), 'Model must implement utils.ifaces.FilesystemFolder to visualize' + \
                                                  'checkpoint losses and upload produced images to cloud'
        # Setup colors
        if colors is None:
            colors = [('#2a9ceb', '#1f77b4'), ('#fa9800', '#fa5622'), ('#8bc34a', '#4caf4f'), ('#9c27b0', '#673ab7'),
                      ('#9e9e9e', '#607d8b')]

        # Download all checkpoints locally
        chkpt_files = {}
        chkpts_dict = self.list_all_checkpoints()
        for epoch, epoch_chkpts in chkpts_dict.items():
            # Check if file is in local filesystem and is a chkpt file
            _files = []
            for _f in epoch_chkpts:
                if isinstance(_f, FilesystemFile) and _f.name.endswith('.pth') and \
                        not _f.name.endswith('__stripped.pth'):
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
                stripped_path = epoch_chkpt.path.replace('.pth', '__stripped.pth')
                if os.path.exists(stripped_path):
                    chkpt_dict = torch.load(stripped_path, map_location='cpu')
                else:
                    chkpt_dict = torch.load(epoch_chkpt.path, map_location='cpu')
                    if 'gen' in chkpt_dict.keys():
                        del chkpt_dict['gen']
                    if 'gen_a_to_b' in chkpt_dict.keys():
                        del chkpt_dict['gen_a_to_b']
                    if 'gen_b_to_a' in chkpt_dict.keys():
                        del chkpt_dict['gen_b_to_a']
                    if 'disc' in chkpt_dict.keys():
                        del chkpt_dict['disc']
                    if 'disc_r' in chkpt_dict.keys():
                        del chkpt_dict['disc_r']
                    if 'disc_a' in chkpt_dict.keys():
                        del chkpt_dict['disc_a']
                    if 'disc_b' in chkpt_dict.keys():
                        del chkpt_dict['disc_b']
                    if extract_dicts:
                        all_keys = []
                        for ki, key_or_keys in enumerate(dict_keys):
                            _keys = (key_or_keys,) if type(key_or_keys) == str else key_or_keys
                            for _k in _keys:
                                if _k not in all_keys:
                                    all_keys.append(_k)
                        stripped_dict = {aki: chkpt_dict[aki] for aki in all_keys if aki in chkpt_dict.keys()}
                        torch.save(stripped_dict, epoch_chkpt.path.replace('.pth', '__stripped.pth'))
                        self.logger.debug(f'{epoch_chkpt.path}: Stripped!')
                        # return after extracting dirs
                        del chkpt_dict
                        gc.collect()
                        time.sleep(1)

                        chkpt_dict = stripped_dict

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
                            # check if there exists a previous value
                            losses_dict[_ii][_k][epoch].append(np.NaN)
                        else:
                            losses_dict[_ii][_k][epoch].append(chkpt_dict[_k])
                del chkpt_dict

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
            _ii = f'img_{ki}'  # ______ # image index
            _idata = losses_dict[_ii]  # image data (e.g. {'gen_loss': {0: [0.1, 0.4, ...], 1: [0.2, ...],...},
            # _________________________ #                   'disc_loss: {....}

            # Create a new figure
            color_index = 0
            fig: plt.Figure
            fig, ax = plt.subplots(figsize=(10, 5), dpi=300, clear=True)
            ax.set_xlabel('epoch')

            # Plot each curve
            curve_key: str
            curve_name: dict
            for curve_name, curve_data in _idata.items():
                # x-axis | y-axis
                curve_x = []
                curve_y = []
                for epoch, epoch_values in curve_data.items():
                    for vi, v in enumerate(epoch_values):
                        curve_x.append(epoch + vi / len(epoch_values))
                        curve_y.append(v)

                # Init y-axis
                if color_index > 0:
                    ax = ax.twinx()
                curve_colors = colors[color_index]
                if len(_idata.keys()) > 1:
                    ax.set_ylabel(curve_name, color=curve_colors[1])
                else:
                    ax.set_ylabel(curve_name)
                ax.tick_params(axis='y')

                # Remove NaNs
                try:
                    curve_y = unnanify(np.array(curve_y))
                except Exception as e:
                    self.logger.critical(str(e))
                    print(curve_y)

                # Plot curve (smooth line + actual points)
                x_new = np.linspace(curve_x[0], curve_x[-1], 300)
                ax.plot(x_new, make_interp_spline(curve_x, curve_y, k=3)(x_new), '-.', color=curve_colors[0],
                        label=curve_name)
                if len(curve_y) > 100:
                    ax.plot(curve_x, curve_y, '.', color=curve_colors[1])
                else:
                    ax.plot(curve_x, curve_y, 'o', color=curve_colors[1])
                color_index += 1

            # Set figure title
            plt_title = f'{" vs. ".join(_keys)}'
            plt_subtitle = f'{filename_suffix.replace("_", " to ").replace("=", ": ").replace(".jpg", "")}'
            fig.suptitle(f'{plt_title}', y=0.97, fontsize=12, fontweight='bold')
            plt.title(f'{plt_subtitle}', pad=10., fontsize=10, )
            fig.legend()
            fig.tight_layout()

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
                # Upload svg
                svg_filepath = filepath.replace(".jpg", ".svg")
                is_update = os.path.exists(svg_filepath)
                if is_update:
                    os.remove(svg_filepath)
                plt.savefig(svg_filepath)
                vis_losses_folder.upload_file(local_filename=svg_filepath, in_parallel=False, is_update=is_update)
                self.logger.debug(f'Loss image from {svg_filepath} uploaded successfully!')
            if preview:
                plt.show()
        return _returns

    def visualize_metrics(self, upload: bool = False, preview: bool = False) -> List[Image]:
        assert isinstance(self, FilesystemModel), 'Model must implement utils.ifaces.FilesystemFolder to visualize' + \
                                                  'metrics and upload produced images to cloud'
        # Init metric lists
        x, fids, iss, prs, rcs, f1s, ssims = ([], [], [], [], [], [], [])
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
                prs.append(metrics['precision'] if not np.isnan(metrics['precision']) else 0)
                rcs.append(metrics['recall'] if not np.isnan(metrics['recall']) else 0)
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
        for metric_name, metric_data in zip(('  FID', '  IS', '  F1', '  PRECISION', '  RECALL', '  SSIM'),
                                            (fids, iss, f1s, prs, rcs, ssims)):
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
            plt.xlabel('epoch')
            # Get PIL image
            pil_img = pltfig_to_pil(plt.gcf())
            _returns.append(pil_img)
            # Save visualization file
            if upload:
                filepath = '{}/{}_{}'.format(vis_metrics_folder.local_root, metric_name.strip().lower(),
                                             filename_suffix)
                is_update = os.path.exists(filepath)
                pil_img.save(filepath)
                self.logger.debug(f'Metric file saved in {filepath}.')
                # Upload to Google Drive
                vis_metrics_folder.upload_file(local_filename=filepath, in_parallel=False, is_update=is_update)
                self.logger.debug(f'Metric file from {filepath} uploaded successfully!')
                # Upload svg
                svg_filepath = filepath.replace(".jpg", ".svg")
                is_update = os.path.exists(svg_filepath)
                if is_update:
                    os.remove(svg_filepath)
                plt.savefig(svg_filepath)
                self.logger.debug(f'Metric SVG file saved in {svg_filepath}.')
                vis_metrics_folder.upload_file(local_filename=svg_filepath, in_parallel=False, is_update=is_update)
                self.logger.debug(f'Loss image from {svg_filepath} uploaded successfully!')
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
        if self.__class__.version() != '1.0':
            model_name_from_class = model_name_from_class + f'_{self.__class__.version()}'
        model_name = os.environ.get(f'NEW_MODEL_NAME__{model_name_from_class}', model_name_from_class)
        if model_name != model_name_from_class:
            self.logger.info(f'Using "{model_name}" instead of "{model_name_from_class}" as model\'s name')
        model_fs_folder = model_fs_folder_or_root if model_fs_folder_or_root.name.endswith(model_name) else \
            model_fs_folder_or_root.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=False)
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
