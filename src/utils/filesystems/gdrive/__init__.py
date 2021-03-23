__all__ = ['colab', 'remote', 'GDriveDataset', 'GDriveModel']

import datetime
import json
import os
from multiprocessing.pool import ApplyResult
from typing import Union, Optional, List, Sequence, Dict

import torch
import yaml
from PIL.Image import Image

from utils.command_line_logger import CommandLineLogger
from utils.filesystems.gdrive.colab import ColabFolder, ColabCapsule, ColabFilesystem, ColabFile
from utils.filesystems.gdrive.remote import GDriveFolder, GDriveCapsule, GDriveFilesystem, GDriveFile
from utils.ifaces import FilesystemDataset, FilesystemModel, FilesystemFolder, Filesystem, FilesystemCapsule, \
    FilesystemFile, ResumableDataLoader


class GDriveDataset(FilesystemDataset):
    """
    GDriveDataset Class:
    This class is used to transfer dataset from Google Drive to local file system and unzip it to be able to use it in
    a data loader afterwards.
    """

    def __init__(self, dataset_fs_folder: FilesystemFolder, zip_filename: str):
        """
        GDriveDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder: a `utils.gdrive.GDriveFolder` instance to interact with dataset 
                                                     folder in Google Drive
        :param (str) zip_filename: the name of dataset's main .zip file (should be inside Google Drive folder root)
        """
        self.dataset_gfolder = dataset_fs_folder
        self.zip_filename = zip_filename
        self.zip_gfile = self.dataset_gfolder.file_by_name(zip_filename)
        assert self.zip_gfile is not None, f'zip_filename={zip_filename} NOT FOUND in Google Drive folder root'

    def fetch_and_unzip(self, in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        if self.is_fetched_and_unzipped():
            if hasattr(self, 'logger') and isinstance(self.logger, CommandLineLogger):
                self.logger.debug('Dataset is fetched and unzipped!')
        return self.dataset_gfolder.download_file(self.zip_gfile, in_parallel=in_parallel,
                                                  show_progress=show_progress, unzip_after=True)

    def is_fetched_and_unzipped(self) -> bool:
        zip_local_filepath = f'{self.dataset_gfolder.local_root}/{self.zip_filename}'
        dataset_local_path = zip_local_filepath.replace('.zip', '')
        return os.path.exists(zip_local_filepath) and os.path.exists(dataset_local_path) and \
               os.path.isfile(zip_local_filepath) and os.path.isdir(dataset_local_path)

    @staticmethod
    def instance(groot_or_capsule_or_fs: Union[FilesystemFolder, FilesystemCapsule, Filesystem],
                 dataset_folder_name: str, zip_filename: str) -> Optional['GDriveDataset']:
        """
        Create ana get a new  `utils.gdrive.GDriveDataset` instance using the given Google Drive root instance, the
        dataset folder name and the zip file name (the entire dataset is assumed to ly inside that zip file).
        :param groot_or_capsule_or_fs: an object to instantiate the Google Drive folder of the dataset
        :param (str) dataset_folder_name: the Google Drive folder name inside which the zipped dataset is placed
        :param (str) zip_filename: the filename (not path) of the zip file containing the actual dataset
        :return:
        """
        # Set Google Drive root folder instance
        if isinstance(groot_or_capsule_or_fs, FilesystemFolder):
            groot = groot_or_capsule_or_fs
        elif isinstance(groot_or_capsule_or_fs, Filesystem):
            groot = groot_or_capsule_or_fs.folder_cls().root(capsule_or_fs=groot_or_capsule_or_fs)
        elif isinstance(groot_or_capsule_or_fs, FilesystemCapsule):
            raise NotImplementedError('not implemented yet for capsule input')
        else:
            raise TypeError('Check groot_or_capsule_or_fs')
        # Find the Google Drive folder instance that corresponds to the dataset with the given folder name
        dataset_gfolder = groot.subfolder_by_name(folder_name=dataset_folder_name, recursive=True)
        if not dataset_gfolder:
            return None
        # Instantiate a GDriveDataset object with the found Google Drive folder instance
        return GDriveDataset(dataset_fs_folder=dataset_gfolder, zip_filename=zip_filename)


class GDriveModel(FilesystemModel):
    """
    GDriveModel Class:
    In the Google Drive the directory structure should be as follows:

    [root]  Models
            ├── model_name=<model_name: str>: checkpoints for model named after `model_name`
            │   ├── Checkpoints (each may contain metrics inside state dict at the "metrics" key)
            │   │   ├── epoch=<epoch: int>
            │   │   │   ├── <step: int or str>.pth
            │   │   │   ├──    ...
            │   │   │   └── <step>.pth
            │   │  ...
            │   │   │
            │   │   └── epoch=<another_epoch>
            │   │       ├── <step>.pth
            │   │       ├──    ...
            │   │       └── <step>.pth
            │   │
            │   ├── Metrics
            │   │   ├── epoch=<epoch: int>
            │   │   │   ├── <step: int or str>.json
            │   │   │   ├──    ...
            │   │   │   └── <step>.json
            │   │  ...
            │   │   │
            │   │   └── epoch=<another_epoch>
            │   │       ├── <step>.json
            │   │       ├──    ...
            │   │       └── <step>.json
            │   │
            │   ├── Visualizations
            │   │   ├── epoch=<epoch: int>
            │   │   │   ├── <step: int or str>.jpg
            │   │   │   ├──    ...
            │   │   │   └── <step>.jpg
            │   │  ...
            │   │   │
            │   │   └── epoch=<another_epoch>
            │   │       ├── <step>.jpg
            │   │       ├──    ...
            │   │       └── <step>.jpg
            │   │
            │   └── Configurations
            │       ├── <config_id: int or str>.yaml
            │       ├──            ...
            │       └── <config_id>.yaml
            │
           ...
            │
            └── model_name=<another_model_name>: checkpoints for model named after `another_model_name`
                    ├── Checkpoints (each may contain metrics inside state dict at the "metrics" key)
                    │   ├── epoch=<epoch>
                    │   │   ├── <step>.pth
                    │   │   ├──    ...
                    │   │   └── <step>.pth
                    │  ...
                    │   │
                    │   └── epoch=<another_epoch>
                    │       ├── <step>.pth
                    │       ├──    ...
                    │       └── <step>.pth
                    │
                    ├── Metrics (each may contain metrics inside state dict at the "metrics" key)
                    │   ├── epoch=<epoch>
                    │   │   ├── <step>.json
                    │   │   ├──    ...
                    │   │   └── <step>.json
                    │  ...
                    │   │
                    │   └── epoch=<another_epoch>
                    │       ├── <step>.json
                    │       ├──    ...
                    │       └── <step>.json
                    │
                    ├── Visualizations
                    │   ├── epoch=<epoch: int>
                    │   │   ├── <step: int or str>.jpg
                    │   │   ├──    ...
                    │   │   └── <step>.jpg
                    │  ...
                    │   │
                    │   └── epoch=<another_epoch>
                    │       ├── <step>.jpg
                    │       ├──    ...
                    │       └── <step>.jpg
                    │
                   ...
                    │
                    └── Configurations
                        ├── <config_id: int or str>.yaml
                        ├──                    ...
                        └── <config_id>.yaml

    Based on this directory structure, this class is used to download/upload model checkpoints to Google Drive and check
    if a checkpoint at a given batch size/step combination is present in the local filesystem.
    """

    def __init__(self, model_fs_folder: FilesystemFolder, logger: CommandLineLogger, model_name: Optional[str] = None,
                 dataset_len: Optional[int] = None):
        """
        GDriveModel class constructor.
        :param (FilesystemFolder) model_fs_folder: a `utils.gdrive.GDriveFolder` instance to interact with model folder
                                                   in local or remote (Google Drive) filesystem
        :param (CommandLineLogger) logger: a `utils.command_line_logger.CommandLineLogger` instance to log events
        :param (str) model_name: the parent model name or `None` to auto-detect from folder name in Google Drive
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        """
        # Save args
        self.logger = logger
        self.gfolder = model_fs_folder
        self.local_chkpts_root = model_fs_folder.local_root
        # Define extra properties
        self.chkpts_gfolder = model_fs_folder.subfolder_by_name_or_create(folder_name='Checkpoints')
        self.metrics_gfolder = model_fs_folder.subfolder_by_name_or_create(folder_name='Metrics')
        self.configurations_gfolder = model_fs_folder.subfolder_by_name_or_create(folder_name='Configurations')
        self.visualizations_fs_folder = model_fs_folder.subfolder_by_name_or_create(folder_name='Visualizations')
        self.model_name = model_name if model_name else \
            model_fs_folder.local_root.split(sep='=', maxsplit=1)[-1]

        # -------------
        # Checkpoints
        # ------------
        # Get all checkpoint folders (i.e. folders with names "epoch=<epoch>")
        self.chkpts_epoch_gfolders = {}
        for _sf in self.chkpts_gfolder.subfolders:
            if _sf.name.startswith('epoch='):
                self.chkpts_epoch_gfolders[int(_sf.name.replace('epoch=', ''))] = _sf
        # No epoch=* subfolders found: create an epoch checkpoint folder with hypothetical epoch=-1
        self.chkpts_epoch_gfolders[-1] = self.chkpts_gfolder

        # ---------
        # Metrics
        # --------
        # Get all metrics folders (i.e. folders with names "epoch=<epoch>" under "Metrics" folder)
        self.metrics_epoch_gfolders = {}
        for _sf in self.metrics_gfolder.subfolders:
            if _sf.name.startswith('epoch='):
                self.metrics_epoch_gfolders[int(_sf.name.replace('epoch=', ''))] = _sf
        # No epoch=* subfolders found: create an epoch metrics folder with hypothetical epoch=-1
        self.metrics_epoch_gfolders[-1] = self.metrics_gfolder
        self.latest_metrics = None
        self.latest_checkpoint_had_metrics = None

        # ----------------
        # Visualizations
        # ---------------
        # Get all visualizations folders (i.e. folders with names "epoch=<epoch>" under "Visualizations" folder)
        self.visualizations_epoch_gfolders = {}
        for _sf in self.visualizations_fs_folder.subfolders:
            if _sf.name.startswith('epoch='):
                self.visualizations_epoch_gfolders[int(_sf.name.replace('epoch=', ''))] = _sf
        # No epoch=* subfolders found: create an epoch visualizations folder with hypothetical epoch=-1
        self.visualizations_epoch_gfolders[-1] = self.visualizations_fs_folder

        # ----------------
        # Configurations
        # ---------------
        self.configurations = None

        # Initialize internal state of step/batch size
        self.step = None
        self.initial_step = 0
        self.epoch = 0
        self._counter = 0
        self.epoch_inc = False
        self.dataset_len = dataset_len

    def gcapture(self, checkpoint: bool = True, metrics: bool = True, visualizations: bool = True,
                 configuration: bool = False, in_parallel: bool = True, show_progress: bool = False,
                 delete_after: bool = False, dataloader: Optional[ResumableDataLoader] = None) \
            -> Union[List[ApplyResult], List[FilesystemFile or None]]:
        """
        Capture the inherited module's current state, save locally and then upload to Google Drive.
        :param (bool) checkpoint: set to True to capture/upload current model state dict & create a new checkpoint in
                                  Google Drive
        :param (bool) metrics: set to True to capture/upload current model metrics Google to Drive
        :param (bool) visualizations: set to True to capture/upload images of latest model's forward pass to Google
                                      Drive
        :param (bool) configuration: set to True to capture/upload current model configuration to Google Drive
        :param (bool) show_progress: set to True to have the capturing/uploading progress displayed in stdout using the
                                     `tqdm` lib
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :param (optional) dataloader: if set and implements `utils.ifaces.ResumableDataLoader`, then the current state
                                      of the dataloader will also be saved in the model checkpoint
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.gdrive.GDriveFile` object if upload completed successfully, `False` with corresponding messages
                 otherwise
        :raises AssertionError: if either `self.step` or `self.epoch` is `None`
        """
        assert not (checkpoint is False and metrics is False and configuration is False)
        _results = []
        # Save model visualizations
        if visualizations:
            # We need at least one forward pass
            assert self.step is not None and self.epoch is not None, 'No forward pass has been performed'
            # Extract current model visualization
            if hasattr(self, 'visualize') and callable(getattr(self, 'visualize')):
                visualization_img = self.visualize()
                if not isinstance(visualization_img, Image):
                    raise AssertionError('visualization_img must be a PIL.Image.Image object')
            else:
                raise NotImplementedError('self must implement utils.ifaces.Visualizable to capture its latest'
                                          ' visualization')
            # Get current epoch and step
            epoch_or_id = self.epoch
            epoch = epoch_or_id if type(epoch_or_id) == int else -1
            # Get new visualizations file name
            new_visualizations_path = self._get_visualizations_filepath(epoch_or_id=epoch_or_id, step=self.step)
            is_update = os.path.exists(new_visualizations_path)
            # Save visualization locally
            visualization_img.save(new_visualizations_path)
            # Upload new visualization file to Google Drive
            #   - ensure visualizations folder exists locally & in Google Drive
            self.ensure_visualizations_fs_folder_exists(epoch=epoch)
            #   - upload local file to Google Drive
            upload_gfolder = self.visualizations_epoch_gfolders[epoch]
            _results.append(upload_gfolder.upload_file(local_filename=os.path.basename(new_visualizations_path),
                                                       delete_after=delete_after, in_parallel=in_parallel,
                                                       show_progress=show_progress, is_update=is_update))
        # Save model checkpoint
        if checkpoint or metrics:
            # Save locally and upload
            assert self.step is not None and self.epoch is not None, 'No forward pass has been performed'
            _results += self.checkpoint_metrics_capture(checkpoint=checkpoint, metrics=metrics, in_parallel=in_parallel,
                                                        delete_after=delete_after, show_progress=show_progress,
                                                        dataloader=dataloader)
        # Save model configuration
        if configuration:
            # Extract current model configuration
            if hasattr(self, 'configuration') and callable(getattr(self, 'configuration')):
                configuration = self.configuration()
            else:
                raise NotImplementedError('self must implement utils.ifaces.Configurable to capture its current current'
                                          ' configuration')
            # Save the extracted configuration and upload to Google Drive
            config_id = self.config_id if hasattr(self, 'config_id') else None
            _results.append(self.save_and_upload_configuration(configuration=configuration, config_id=config_id,
                                                               delete_after=delete_after, in_parallel=in_parallel,
                                                               show_progress=show_progress))
        return _results

    def gforward(self, batch_size: int = None) -> None:
        """
        Function to be triggered on inherited model's forward pass to set step and epoch internally.
        :param (optional) batch_size: the current batch size (to update internal samples counter and therefore keep
                                      track of the current epoch and step)
        """
        if self.dataset_len is None:
            raise AttributeError('self.dataset_len is None: need to re-implement some methods to allow this...')
        # Update current step (ever-growing counter)
        self.step = self.step + 1 if self.step else 1
        self.initial_step += 1
        # Update current epoch
        if self.epoch_inc:
            self.epoch = self.epoch + 1
            self._counter = 0
            self.epoch_inc = False
        # Update number of samples seen
        self._counter += batch_size
        # Check if enc of epoch reached
        assert self._counter <= self.dataset_len, f'self._counter={self._counter} > self.dataset_len={self.dataset_len}'
        if self._counter == self.dataset_len:
            self.epoch_inc = True
            self.initial_step = 0
        # Debug info
        # self.logger.debug(f'self.dataset_len={self.dataset_len}, self._counter={self._counter}, '
        #                   f'self.step={self.step}, self.epoch={self.epoch}, self.epoch_inc={self.epoch_inc}')

    def load_gforward_state(self, state: dict):
        """
        Loads running statistics (e.g. counters) from the model checkpoint.
        :param (dict) state: a `dict` object containing the keys corresponding to instance properties used by gforward()
        """
        self.step = state['step']
        self.initial_step = state['initial_step']
        self.epoch = state['epoch']
        self.epoch_inc = state['epoch_inc']
        self._counter = state['_counter']
        self.logger.debug(f'Loading gforward state: {str(state)}')

    #
    # ------------------------------
    #  Model Checkpoints & Metrics
    # -----------------------------
    #
    # Below, are the methods to capture, save, upload and download model checkpoints to cloud storage.
    #

    def checkpoint_metrics_capture(self, checkpoint: bool, metrics: bool, delete_after: bool = False,
                                   dataloader: Optional[ResumableDataLoader] = None, in_parallel: bool = False,
                                   show_progress: bool = False) -> List[ApplyResult or FilesystemFile or None]:
        """
        Save locally and upload to Google Drive current model checkpoint and/or evaluation metrics.
        :param (bool) checkpoint: set to True to save & upload model checkpoint .pth (a.k.a. state dict)
        :param (bool) metrics: set to True to save & upload model metrics .json file
        :param (bool) delete_after: set to True to delete local file(s) after successful upload
        :param (bool) in_parallel: set to True to run the file(s) uploading in a separate thread
        :param (bool) show_progress: set to True to show progress while generating metrics & uploading local file(s)
        :param (optional) dataloader: if set and implements `utils.ifaces.ResumableDataLoader`, then the current state
                                      of the dataloader will also be saved in the model checkpoint
        :return: a `list` of upload results, see `utils.ifaces.Filesystem::upload_local_file()`
        """
        # Get state dict
        state_dict = None
        metrics_dict = None
        if checkpoint:
            if hasattr(self, 'state_dict') and callable(getattr(self, 'state_dict')):
                state_dict = self.state_dict()
                state_dict['gforward'] = {
                    'step': self.step,
                    'initial_step': self.initial_step,
                    'epoch': self.epoch,
                    '_counter': self._counter,
                    'epoch_inc': self.epoch_inc,
                }
                # Check for dataloader
                if dataloader:
                    assert isinstance(dataloader, ResumableDataLoader), 'dataloader must implement ResumableDataLoader'
                    state_dict['dataloader'] = dataloader.get_state()
            else:
                raise NotImplementedError('self.state_dict() is not defined')
        # Get evaluation metrics
        if metrics:
            if hasattr(self, 'evaluate') and callable(getattr(self, 'evaluate')):
                metrics_dict = self.evaluate(show_progress=show_progress)
                # Save model metrics in checkpoint
                if state_dict:
                    state_dict['metrics'] = metrics_dict
                # Save model metrics in instance
                self.latest_metrics = metrics_dict
                self.latest_checkpoint_had_metrics = True
            else:
                raise NotImplementedError('self.evaluate() is not defined')
        else:
            self.latest_checkpoint_had_metrics = False
        return self.save_and_upload_checkpoint(state_dict=state_dict, metrics_dict=metrics_dict, epoch_or_id=self.epoch,
                                               step=self.step, delete_after=delete_after, in_parallel=in_parallel,
                                               show_progress=show_progress)

    def download_checkpoint(self, epoch_or_id: Union[int, str], step: Optional[int] = None, in_parallel: bool = False,
                            show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Get correct epoch & step
        epoch = epoch_or_id if type(epoch_or_id) == int else -1
        # Ensure folder exists locally & in Google Drive
        self.ensure_chkpts_gfolder_exists(epoch=epoch)
        # Get the correct GoogleDrive folder to search for checkpoint files
        chkpts_gfolder = self.chkpts_epoch_gfolders[epoch]

        # Search for the checkpoint in the list with all model checkpoints inside checkpoints folder for given batch
        chkpt_filepath = self._get_checkpoint_filepath(epoch_or_id=epoch_or_id, step=step)
        chkpt_filename = os.path.basename(chkpt_filepath)
        for _f in chkpts_gfolder.files:
            if _f.name == chkpt_filename:
                # Checkpoint file found!
                self.logger.debug(f'Checkpoint file found. Downloading now at: {_f.path}')
                return _f.download(in_parallel=in_parallel, show_progress=show_progress)
        # If reached here, no checkpoint files matching given step & batch size could be found
        raise FileNotFoundError(f'No checkpoint file could be found inside "{chkpts_gfolder.name}" matching ' +
                                f'epoch="{epoch}" and step={step if type(epoch_or_id) is int else epoch_or_id}')

    def ensure_chkpts_gfolder_exists(self, epoch: Optional[int] = None) -> None:
        """
        Checks if folder for given batch size exists locally as well as in Google Drive
        :param (int or None) epoch: the folder should be named "epoch=<epoch>"; this is where this parameter is used.
                                    if :attr:`epoch` is `None`, then the root Checkpoints folder is considered
        """
        # Convert "epoch=None" folder name to epoch = -1 (thus targeting root checkpoints folder), since None cannot
        # be a valid dict key
        if epoch is None:
            epoch = -1
        # Check instance for folder of given epoch
        if epoch not in self.chkpts_epoch_gfolders.keys():
            # Folder for given batch size does not exist, create a new folder now and save in instance's dict
            # This will also create folder locally
            self.chkpts_epoch_gfolders[epoch] = self.chkpts_gfolder.create_subfolder(folder_name=f'epoch={epoch}',
                                                                                     force_create_local=True)

    def ensure_metrics_gfolder_exists(self, epoch: Optional[int] = None) -> None:
        """
        Checks if metrics folder for given batch size exists locally as well as in Google Drive.
        :param (int or None) epoch: the folder should be named "epoch=<epoch>"; this is where this parameter is used.
                                    if :attr:`epoch` is `None`, then the root Checkpoints folder is considered
        """
        # Convert "epoch=None" folder name to epoch = -1 (thus targeting root checkpoints folder), since None cannot
        # be a valid dict key
        if epoch is None:
            epoch = -1
        # Check instance for folder of given epoch
        if epoch not in self.metrics_epoch_gfolders.keys():
            # Folder for given batch size does not exist, create a new folder now and save in instance's dict
            # This will also create folder locally
            self.metrics_epoch_gfolders[epoch] = \
                self.metrics_gfolder.create_subfolder(folder_name=f'epoch={epoch}', force_create_local=True)

    def ensure_visualizations_fs_folder_exists(self, epoch: Optional[int] = None) -> None:
        """
        Checks if visualizations folder for given batch size exists locally as well as in Google Drive.
        :param (int or None) epoch: the folder should be named "epoch=<epoch>"; this is where this parameter is used.
                                    if :attr:`epoch` is `None`, then the root Checkpoints folder is considered
        """
        # Convert "epoch=None" folder name to epoch = -1 (thus targeting root checkpoints folder), since None cannot
        # be a valid dict key
        if epoch is None:
            epoch = -1
        # Check instance for folder of given epoch
        if epoch not in self.visualizations_epoch_gfolders.keys():
            # Folder for given batch size does not exist, create a new folder now and save in instance's dict
            # This will also create folder locally
            self.visualizations_epoch_gfolders[epoch] = \
                self.visualizations_fs_folder.create_subfolder(folder_name=f'epoch={epoch}', force_create_local=True)

    def fetch_checkpoint(self, epoch_or_id: Union[int, str], step: Optional[int] = None) -> str or False:
        # Check if latest checkpoint is requested
        if 'latest' == epoch_or_id:
            return self.fetch_latest_checkpoint(epoch=None)
        # Check if checkpoint file exists in local filesystem
        local_filepath = self.is_checkpoint_fetched(epoch_or_id=epoch_or_id, step=step)
        if local_filepath:
            return local_filepath
        # Download checkpoint file from Google Drive
        if self.download_checkpoint(epoch_or_id=epoch_or_id, step=step, in_parallel=False, show_progress=True):
            _path = self.is_checkpoint_fetched(epoch_or_id=epoch_or_id, step=step)
            if not _path:
                return False
            # Update internal state
            if type(epoch_or_id) is int:
                self.epoch = epoch_or_id
            if step and type(step) is int:
                self.step = step
            return _path
        # If reaches here, file could not be downloaded, probably due to an unidentified error
        raise ValueError('self.download_checkpoint returned False')

    def fetch_latest_checkpoint(self, epoch: Optional[int] = None) -> str or False:
        # Format args
        if epoch is None:
            epoch = sorted(self.chkpts_epoch_gfolders.keys(), reverse=True)[0]
        epoch = -1 if epoch is None else epoch
        # Get all checkpoint files under given epoch folder
        chkpts = self.list_checkpoints(epoch=epoch, only_keys=('title',))
        if len(chkpts) == 0:
            raise FileNotFoundError(f'No checkpoint files could be found matching epoch="{epoch}"')
        # Sort filenames in descending order
        chkpts = sorted(chkpts, key=lambda _d: _d['title'], reverse=True)
        # Take the first and download it locally
        latest_step = chkpts[0]['title'].replace('.pth', '')
        step = int(latest_step) if epoch != -1 and latest_step.isdigit() else None
        return self.fetch_checkpoint(epoch_or_id=latest_step if epoch == -1 else epoch, step=step)

    def _get_checkpoint_filepath(self, epoch_or_id: Union[int or str] = None, step: Optional[int] = None) -> str:
        """
        Get the absolute path to the local checkpoint file at given :attr:`epoch` and :attr:`step` or whose filename is
        ":attr:`id`.pth".
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (int or str) step: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :return:
        """
        # Check & correct given args
        if type(epoch_or_id) == str:
            epoch = -1
            assert step is None, f'provided checkpoint id (epoch_or_id={epoch_or_id}) is mutually exclusive with ' + \
                                 f'step attribute (step={step})'
            chkpt_id = epoch_or_id
        else:
            epoch = epoch_or_id
            assert step is not None, f'provided epoch number (epoch_or_id={epoch}) but step=None. No way to find file'
            chkpt_id = str(step).zfill(10)
        # Find containing folder
        self.ensure_chkpts_gfolder_exists(epoch=epoch)
        chkpts_gfolder = self.chkpts_epoch_gfolders[epoch]
        chkpts_gfolder.ensure_local_root_exists()
        # Format and return
        return f'{chkpts_gfolder.local_root}/{chkpt_id}.pth'

    def _get_metrics_filepath(self, epoch_or_id: Union[int or str] = None, step: Optional[int] = None) -> str:
        """
        Get the absolute path to the local metrics file at given :attr:`epoch` and :attr:`step` or whose filename is
        ":attr:`id`.json".
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (int or str) step: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :return:
        """
        # Check & correct given args
        if type(epoch_or_id) == str:
            epoch = -1
            assert step is None, f'provided metrics id (epoch_or_id={epoch_or_id}) is mutually exclusive with ' + \
                                 f'step attribute (step={step})'
            chkpt_id = epoch_or_id
        else:
            epoch = epoch_or_id
            assert step is not None, f'provided epoch number (epoch_or_id={epoch}) but step=None. No way to find file'
            chkpt_id = str(step).zfill(10)
        # Find containing folder
        self.ensure_metrics_gfolder_exists(epoch=epoch)
        metrics_gfolder = self.metrics_epoch_gfolders[epoch]
        metrics_gfolder.ensure_local_root_exists()
        # Format and return
        return f'{metrics_gfolder.local_root}/{chkpt_id}.json'

    def _get_visualizations_filepath(self, epoch_or_id: Union[int or str] = None, step: Optional[int] = None) -> str:
        """
        Get the absolute path to the local visualizations file at given :attr:`epoch` and :attr:`step` or whose filename
        is ":attr:`id`.jpg".
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (int or str) step: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :return: absolute file path as a `str object`
        """
        # Check & correct given args
        if type(epoch_or_id) == str:
            epoch = -1
            assert step is None, f'provided visualizations id (epoch_or_id={epoch_or_id}) is mutually exclusive ' + \
                                 f'with step attribute (step={step})'
            chkpt_id = epoch_or_id
        else:
            epoch = epoch_or_id
            assert step is not None, f'provided epoch number (epoch_or_id={epoch}) but step=None. No way to find file'
            chkpt_id = str(step).zfill(10)
        # Find containing folder
        self.ensure_visualizations_fs_folder_exists(epoch=epoch)
        visualizations_fs_folder = self.visualizations_epoch_gfolders[epoch]
        visualizations_fs_folder.ensure_local_root_exists()
        # Format and return
        return f'{visualizations_fs_folder.local_root}/{chkpt_id}.jpg'

    def is_checkpoint_fetched(self, epoch_or_id: Union[int, str], step: Optional[int] = None) -> str or False:
        local_filepath = self._get_checkpoint_filepath(epoch_or_id=epoch_or_id, step=step)
        return local_filepath if os.path.exists(local_filepath) and os.path.isfile(local_filepath) \
            else False

    # noinspection DuplicatedCode
    def list_checkpoints(self, epoch: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[GDriveFile or ColabFile or dict]:
        # Check & correct given args
        if epoch is None:
            epoch = -1
        # Get the correct GoogleDrive folder to list the checkpoint files that are inside
        if epoch not in self.chkpts_epoch_gfolders.keys():
            raise FileNotFoundError(f'epoch="{epoch}" not found in self.chkpts_epoch_gfolders.keys()')
        # Get checkpoint files list and filter it if only_keys attribute is set
        chkpt_files_list = self.chkpts_epoch_gfolders[epoch].files
        return chkpt_files_list if not only_keys else \
            [dict((k, _f[k]) for k in only_keys) for _f in chkpt_files_list]

    # noinspection DuplicatedCode
    def list_metrics(self, epoch: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[GDriveFile or ColabFile or dict]:
        # Check & correct given args
        if epoch is None:
            epoch = -1
        # Get the correct GoogleDrive folder to list the metric files that are inside
        if epoch not in self.metrics_epoch_gfolders.keys():
            raise FileNotFoundError(f'epoch="{epoch}" not found in self.metrics_epoch_gfolders.keys()')
        # Get metric files list and filter it if only_keys attribute is set
        metric_files_list = self.metrics_epoch_gfolders[epoch].files
        return metric_files_list if not only_keys else \
            [dict((k, _f[k]) for k in only_keys) for _f in metric_files_list]

    def list_all_checkpoints(self, only_keys: Optional[Sequence[str]] = None) \
            -> Dict[int, List[GDriveFile or ColabFile or dict]]:
        _return_dict = {}
        for _epoch in self.chkpts_epoch_gfolders.keys():
            _epoch_chkpts_list = self.list_checkpoints(epoch=_epoch, only_keys=only_keys)
            if len(_epoch_chkpts_list) > 0:
                _return_dict[_epoch] = _epoch_chkpts_list
        return _return_dict

    def list_all_metrics(self, only_keys: Optional[Sequence[str]] = None) \
            -> Dict[int, List[GDriveFile or ColabFile or dict]]:
        _return_dict = {}
        for _epoch in sorted(self.metrics_epoch_gfolders.keys(), key=lambda _k: int(_k)):
            _epoch_metrics_list = self.list_metrics(epoch=_epoch, only_keys=only_keys)
            if len(_epoch_metrics_list) > 0:
                _return_dict[_epoch] = _epoch_metrics_list
        return _return_dict

    def save_and_upload_checkpoint(self, state_dict: dict, epoch_or_id: Union[int, str], step: Optional[int] = None,
                                   metrics_dict: Optional[dict] = None, delete_after: bool = False,
                                   in_parallel: bool = False, show_progress: bool = False) \
            -> Union[List[ApplyResult], List[FilesystemFile or None]]:
        _results = []
        # Get correct epoch number or -1 if no epoch info is saved
        epoch = epoch_or_id if type(epoch_or_id) == int else -1
        # Upload state dict in Google Drive (under "Checkpoints" folder)
        if state_dict:
            # Get new checkpoint file name
            new_chkpt_path = self._get_checkpoint_filepath(epoch_or_id=epoch_or_id, step=step)
            is_update = os.path.exists(new_chkpt_path)
            # Save state_dict locally
            torch.save(state_dict, new_chkpt_path)
            # Upload new checkpoint file to Google Drive
            _results.append(self.upload_checkpoint(chkpt_filename=new_chkpt_path, epoch=epoch,
                                                   delete_after=delete_after, in_parallel=in_parallel,
                                                   show_progress=show_progress, is_update=is_update))
        # Upload metrics dict in Google Drive (under "Metrics" folder)
        if metrics_dict:
            # Get new metrics file name
            new_metrics_path = self._get_metrics_filepath(epoch_or_id=epoch_or_id, step=step)
            is_update = os.path.exists(new_metrics_path)
            # Save metrics_dict locally
            with open(new_metrics_path, 'w') as json_fp:
                json.dump(metrics_dict, json_fp, indent=4)
            # Upload new metrics file to Google Drive
            #   - ensure metrics folder exists locally & in Google Drive
            self.ensure_metrics_gfolder_exists(epoch=epoch)
            #   - upload local file to Google Drive
            upload_gfolder = self.metrics_epoch_gfolders[epoch]
            _results.append(upload_gfolder.upload_file(local_filename=os.path.basename(new_metrics_path),
                                                       delete_after=delete_after, in_parallel=in_parallel,
                                                       show_progress=show_progress, is_update=is_update))
        return _results

    def upload_checkpoint(self, chkpt_filename: str, epoch: Optional[int] = None, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, FilesystemFile, None]:
        # Check if needed to create the folder in Google Drive before uploading
        if epoch is None:
            epoch = -1
        # Ensure folder exists locally & in Google Drive
        self.ensure_chkpts_gfolder_exists(epoch=epoch)
        # Upload local file to Google Drive
        upload_gfolder = self.chkpts_epoch_gfolders[epoch]
        return upload_gfolder.upload_file(local_filename=os.path.basename(chkpt_filename), delete_after=delete_after,
                                          in_parallel=in_parallel, show_progress=show_progress, is_update=is_update)

    #
    # ------------------------------------
    #  Model Configurations
    # ---------------------------------
    #
    # Below, are the methods to capture, save, upload and download model configurations to/from cloud storage.
    #

    def download_configuration(self, config_id: Union[int, str], in_parallel: bool = False,
                               show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Check & correct given args
        config_id_str = config_id if isinstance(config_id, str) else str(config_id).zfill(10)
        # Search for the checkpoint in the list with all model checkpoints inside checkpoints folder for given batch
        config_filename = f'{config_id_str}.yaml'
        conf_gfile = self.configurations_gfolder.file_by_name(filename=config_filename)
        if conf_gfile:
            return conf_gfile.download(in_parallel=in_parallel, show_progress=show_progress)
        # If reached here, no checkpoint files matching given step & batch size could be found
        raise FileNotFoundError(f'No configuration file could be found inside "{self.configurations_gfolder.name}" '
                                f'matching config_id="{config_id_str}"')

    def fetch_configuration(self, config_id: Union[int, str]) -> str or False:
        # Check if checkpoint file exists in local filesystem
        local_filepath = self.is_configuration_fetched(config_id=config_id)
        if local_filepath:
            return local_filepath
        # Download checkpoint file from Google Drive
        if self.download_configuration(config_id=config_id, in_parallel=False, show_progress=True):
            return self.is_configuration_fetched(config_id=config_id)
        # If reaches here, file could not be downloaded, probably due to an unidentified error
        raise ValueError('self.download_configuration returned False')

    def is_configuration_fetched(self, config_id: Union[int, str]) -> str or False:
        # Check & correct given args
        config_id_str = config_id if isinstance(config_id, str) else str(config_id).zfill(10)
        # Search for the configuration in the list with all model configurations inside Configurations folder for
        # given configuration id
        local_filepath = f'{self.configurations_gfolder.local_root}/{config_id_str}.yaml'
        return local_filepath if os.path.exists(local_filepath) and os.path.isfile(local_filepath) \
            else False

    def list_configurations(self, only_keys: Optional[Sequence] = None) -> List[GDriveFile or ColabFile or dict]:
        # Get configuration files list and filter it if only_keys attribute is set
        config_files_list = self.configurations_gfolder.files
        return config_files_list if not only_keys else \
            [dict((k, _f[k]) for k in only_keys) for _f in config_files_list]

    def save_and_upload_configuration(self, configuration: dict, config_id: Optional[str or int] = None,
                                      delete_after: bool = False, in_parallel: bool = False,
                                      show_progress: bool = False) -> ApplyResult or FilesystemFile or None:
        # Check & correct given args
        if config_id:
            config_id_str = config_id if isinstance(config_id, str) else str(config_id).zfill(10)
        else:
            config_id_str = str(int(datetime.datetime.now().timestamp()))
        # Get new checkpoint file name
        new_config_filepath = f'{self.configurations_gfolder.local_root}/{config_id_str}.yaml'
        is_update = os.path.exists(new_config_filepath)
        # Save config file locally
        self.configurations_gfolder.ensure_local_root_exists()
        with open(new_config_filepath, 'w') as yaml_fp:
            yaml.dump(configuration, yaml_fp)
        # Upload new configuration file to Google Drive
        return self.upload_configuration(config_filename=new_config_filepath, delete_after=delete_after,
                                         in_parallel=in_parallel, show_progress=show_progress, is_update=is_update)

    def upload_configuration(self, config_filename: str, delete_after: bool = False, in_parallel: bool = False,
                             show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, GDriveFile or ColabFile or None]:
        # Upload local file to Google Drive
        return self.configurations_gfolder.upload_file(local_filename=os.path.basename(config_filename),
                                                       delete_after=delete_after, in_parallel=in_parallel,
                                                       show_progress=show_progress, is_update=is_update)
