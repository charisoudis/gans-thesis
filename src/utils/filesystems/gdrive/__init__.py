__all__ = ['colab', 'remote', 'GDriveDataset', 'GDriveModel']

import datetime
import os
from multiprocessing.pool import ApplyResult
from typing import Union, Optional, List, Sequence, Dict

import torch
import yaml

from utils.command_line_logger import CommandLineLogger
from utils.filesystems.gdrive.colab import ColabFolder, ColabCapsule, ColabFilesystem, ColabFile
from utils.filesystems.gdrive.remote import GDriveFolder, GDriveCapsule, GDriveFilesystem, GDriveFile
from utils.ifaces import FilesystemDataset, FilesystemModel, FilesystemFolder, Filesystem, FilesystemCapsule, \
    FilesystemFile


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
        return self.dataset_gfolder.download_file(self.zip_gfile, in_parallel=in_parallel,
                                                  show_progress=show_progress, unzip_after=True)

    def is_fetched_and_unzipped(self) -> bool:
        zip_local_filepath = f'{self.dataset_gfolder.local_root}/{self.zip_filename}'
        dataset_local_path = zip_local_filepath.replace('.zip', '')
        return os.path.exists(zip_local_filepath) and os.path.exists(dataset_local_path) and \
               os.path.isfile(zip_local_filepath) and os.path.isdir(dataset_local_path)

    @staticmethod
    def instance(groot_or_capsule_or_fs: Union[FilesystemFolder, Filesystem, FilesystemCapsule],
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
        groot = None
        if isinstance(groot_or_capsule_or_fs, FilesystemFolder):
            groot = groot_or_capsule_or_fs
        elif isinstance(groot_or_capsule_or_fs, Filesystem) or isinstance(groot_or_capsule_or_fs, FilesystemCapsule):
            if isinstance(groot_or_capsule_or_fs, GDriveFilesystem) \
                    or isinstance(groot_or_capsule_or_fs, GDriveCapsule):
                groot = GDriveFolder.root(capsule_or_fs=groot_or_capsule_or_fs)
            elif isinstance(groot_or_capsule_or_fs, ColabFilesystem) \
                    or isinstance(groot_or_capsule_or_fs, ColabCapsule):
                groot = ColabFolder.root(capsule_or_fs=groot_or_capsule_or_fs)
            else:
                raise NotImplementedError
        if not groot:
            raise ValueError('Could not instantiate groot')
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
            │   │   ├── batch_size=<batch_size: int or None>
            │   │   │   ├── <step: int or str>.pth
            │   │   │   ├──    ...
            │   │   │   └── <step>.pth
            │   │  ...
            │   │   │
            │   │   └── batch_size=<another_batch_size>
            │   │       ├── <step>.pth
            │   │       ├──    ...
            │   │       └── <step>.pth
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
                    │   ├── batch_size=<batch_size>
                    │   │   ├── <step>.pth
                    │   │   ├──    ...
                    │   │   └── <step>.pth
                    │  ...
                    │   │
                    │   └── batch_size=<another_batch_size>
                    │       ├── <step>.pth
                    │       ├──    ...
                    │       └── <step>.pth
                   ...
                    │
                    └── Configurations
                        ├── <config_id: int or str>.yaml
                        ├──                    ...
                        └── <config_id>.yaml

    Based on this directory structure, this class is used to download/upload model checkpoints to Google Drive and check
    if a checkpoint at a given batch size/step combination is present in the local filesystem.
    """

    def __init__(self, model_fs_folder: FilesystemFolder, model_name: Optional[str] = None):
        """
        GDriveModel class constructor.
        :param (FilesystemFolder) model_fs_folder: a `utils.gdrive.GDriveFolder` instance to interact with model folder
                                                   in local or remote (Google Drive) filesystem
        :param (str) model_name: the parent model name or `None` to auto-detect from folder name in Google Drive
        """
        self.logger = CommandLineLogger(log_level='info', name=self.__class__.__name__)
        # Save args
        self.gfolder = model_fs_folder
        self.local_chkpts_root = model_fs_folder.local_root
        # Define extra properties
        self.chkpts_gfolder = model_fs_folder.subfolder_by_name(folder_name='Checkpoints')
        self.metrics_gfolder = model_fs_folder.subfolder_by_name(folder_name='Metrics')
        self.configurations_gfolder = model_fs_folder.subfolder_by_name_or_create(folder_name='Configurations')
        self.model_name = model_name if model_name else \
            model_fs_folder.local_root.split(sep='=', maxsplit=1)[-1]

        # -------------
        # Checkpoints
        # ------------
        # Get all checkpoint folders (i.e. folders with names "batch_size=<batch_size>")
        self.chkpts_batch_gfolders = {}
        for _sf in self.chkpts_gfolder.subfolders:
            if _sf.name.startswith('batch_size='):
                self.chkpts_batch_gfolders[int(_sf.name.replace('batch_size=', ''))] = _sf
        # No batch_size=* subfolders found: create a batch checkpoint folder with hypothetical batch_size=-1
        self.chkpts_batch_gfolders[-1] = self.chkpts_gfolder
        # Initialize internal state of step/batch size
        self.step = None
        self.batch_size = None

        # ----------------
        # Configurations
        # ---------------
        self.configurations = None

    def gcapture(self, checkpoint: bool = True, metrics: bool = True, configuration: bool = False,
                 in_parallel: bool = True, show_progress: bool = False, delete_after: bool = False) \
            -> Union[List[ApplyResult], List[FilesystemFile or None]]:
        """
        Capture the inherited module's current state, save locally and then upload to Google Drive.
        :param (bool) checkpoint: set to True to capture/upload current model state dict & create a new checkpoint in
                                  Google Drive
        :param (bool) metrics: set to True to capture/upload current model metrics Google to Drive
        :param (bool) configuration: set to True to capture/upload current model configuration to Google Drive
        :param (bool) show_progress: set to True to have the capturing/uploading progress displayed in stdout using the
                                     `tqdm` lib
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.gdrive.GDriveFile` object if upload completed successfully, `False` with corresponding messages
                 otherwise
        :raises AssertionError: if either `self.step` or `self.batch_size` is `None`
        """
        assert not (checkpoint is False and metrics is False and configuration is False)
        _returns = []
        # Save model checkpoint
        if checkpoint:
            # Get state dict
            if hasattr(self, 'state_dict') and callable(getattr(self, 'state_dict')):
                state_dict = self.state_dict()
            else:
                raise NotImplementedError('self.state_dict() is not defined')
            # Save model metrics in checkpoint
            if metrics:
                if hasattr(self, 'evaluate') and callable(getattr(self, 'evaluate')):
                    metrics_dict = self.evaluate(show_progress=show_progress)
                    state_dict['metrics'] = metrics_dict
                else:
                    raise NotImplementedError('self.evaluate() is not defined')
            # Save locally and upload
            assert self.step is not None and self.batch_size is not None, 'No forward pass has been performed'
            _returns.append(self.save_and_upload_checkpoint(state_dict=state_dict, step=self.step,
                                                            batch_size=self.batch_size, delete_after=delete_after,
                                                            in_parallel=in_parallel, show_progress=show_progress))
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
            _returns.append(self.save_and_upload_configuration(configuration=configuration, config_id=config_id,
                                                               delete_after=delete_after, in_parallel=in_parallel,
                                                               show_progress=not in_parallel))
        return _returns

    #
    # ------------------------------------
    #  Model Checkpoints
    # ---------------------------------
    #
    # Below, are the methods to capture, save, upload and download model checkpoints to cloud storage.
    #

    def gforward(self, batch_size: Optional[int] = None) -> None:
        """
        Function to be triggered on inherited model's forward pass to set step and batch_size internally.
        :param (optional) batch_size: if `self.batch_size` is `None`, then this will be used to initialize it
        """
        self.step = 1 if not self.step else \
            self.step + 1
        if not self.batch_size:
            self.batch_size = batch_size

    def download_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None,
                            in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Check & correct given args
        if batch_size is None:
            batch_size = -1
        step_str = step if isinstance(step, str) else str(step).zfill(10)
        # Ensure folder exists locally & in Google Drive
        self.ensure_chkpts_gfolder_exists(batch_size=batch_size)
        # Get the correct GoogleDrive folder to search for checkpoint files
        chkpts_gfolder = self.chkpts_batch_gfolders[batch_size]

        # Search for the checkpoint in the list with all model checkpoints inside checkpoints folder for given batch
        chkpt_filename = f'{step_str}.pth'
        for _f in chkpts_gfolder.files:
            if _f.name == chkpt_filename:
                # Checkpoint file found!
                return _f.download(in_parallel=in_parallel, show_progress=show_progress)
        # If reached here, no checkpoint files matching given step & batch size could be found
        raise FileNotFoundError(f'No checkpoint file could be found inside "{chkpts_gfolder.name}" matching ' +
                                f'step="{step_str}"' + '' if batch_size == -1 else f' and batch_size="{batch_size}"')

    def ensure_chkpts_gfolder_exists(self, batch_size: Optional[int] = None) -> None:
        """
        Checks if folder for given batch size exists locally as well as in Google Drive
        :param (int or None) batch_size: the folder should be named "batch_size=<batch_size>"; this is where this
                                         parameter is used
        """
        # Convert "batch_size=None" folder name to batch_size = -1, since None cannot be a valid dict key
        if batch_size is None:
            batch_size = -1
        # Check instance for folder of given batch_size
        if batch_size not in self.chkpts_batch_gfolders.keys():
            # Folder for given batch size does not exist, create a new folder now and save in instance's dict
            # This will also create folder locally
            self.chkpts_batch_gfolders[batch_size] = \
                self.chkpts_gfolder.create_subfolder(folder_name=f'batch_size={str(batch_size)}',
                                                     force_create_local=True)

    def fetch_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
        if 'latest' == step:
            return self.fetch_latest_checkpoint(batch_size=batch_size)
        # Check if checkpoint file exists in local filesystem
        local_filepath = self.is_checkpoint_fetched(step=step, batch_size=batch_size)
        if local_filepath:
            return local_filepath
        # Download checkpoint file from Google Drive
        if self.download_checkpoint(step=step, batch_size=batch_size, in_parallel=False, show_progress=True):
            return self.is_checkpoint_fetched(step=step, batch_size=batch_size)
        # If reaches here, file could not be downloaded, probably due to an unidentified error
        raise ValueError('self.download_checkpoint returned False')

    def fetch_latest_checkpoint(self, batch_size: Optional[int] = None) -> str or False:
        chkpts = self.list_checkpoints(batch_size=batch_size, only_keys=('title',))
        if len(chkpts) == 0:
            raise FileNotFoundError(f'No latest checkpoint file could be found matching batch_size="{batch_size}"')
        chkpts = sorted(chkpts, key=lambda _d: _d['title'], reverse=True)
        latest_step = chkpts[0]['title'].replace('.pth', '')
        return self.fetch_checkpoint(step=int(latest_step) if latest_step.isdigit() else latest_step,
                                     batch_size=batch_size)

    def _get_checkpoint_filepath(self, step: Union[int, str], batch_size: Optional[int] = None):
        """
        Get the absolute path to the local checkpoint file at given :attr:`step` and :attr:`batch_size`.
        :param step: TODO
        :param batch_size:
        :return:
        """
        # Check & correct given args
        if batch_size is None:
            batch_size = -1
        step_str = step if isinstance(step, str) else str(step).zfill(10)
        # Find containing folder
        if batch_size not in self.chkpts_batch_gfolders.keys():
            self.ensure_chkpts_gfolder_exists(batch_size=batch_size)
        chkpts_gfolder = self.chkpts_batch_gfolders[batch_size]
        chkpts_gfolder.ensure_local_root_exists()
        # Format and return
        return f'{chkpts_gfolder.local_root}/{step_str}.pth'

    def is_checkpoint_fetched(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
        local_filepath = self._get_checkpoint_filepath(step=step, batch_size=batch_size)
        return local_filepath if os.path.exists(local_filepath) and os.path.isfile(local_filepath) \
            else False

    def list_checkpoints(self, batch_size: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[GDriveFile or ColabFile or dict]:
        # Check & correct given args
        if batch_size is None:
            batch_size = -1
        # Get the correct GoogleDrive folder to list the checkpoint files that are inside
        if batch_size not in self.chkpts_batch_gfolders.keys():
            raise FileNotFoundError(f'batch_size="{batch_size}" not found in self.chkpts_batch_gfolders.keys()')
        # Get checkpoint files list and filter it if only_keys attribute is set
        chkpt_files_list = self.chkpts_batch_gfolders[batch_size].files
        return chkpt_files_list if not only_keys else \
            [dict((k, _f[k]) for k in only_keys) for _f in chkpt_files_list]

    def list_all_checkpoints(self, only_keys: Optional[Sequence[str]] = None) \
            -> Dict[int, List[GDriveFile or ColabFile or dict]]:
        _return_dict = {}
        for batch_size in self.chkpts_batch_gfolders.keys():
            batch_chkpts_list = self.list_checkpoints(batch_size=batch_size, only_keys=only_keys)
            if len(batch_chkpts_list) > 0:
                _return_dict[batch_size] = batch_chkpts_list
        return _return_dict

    def save_and_upload_checkpoint(self, state_dict: dict, step: Union[int, str], batch_size: Optional[int] = None,
                                   delete_after: bool = False, in_parallel: bool = False,
                                   show_progress: bool = False) -> Union[ApplyResult, GDriveFile or ColabFile or None]:
        # Get new checkpoint file name
        new_chkpt_path = self._get_checkpoint_filepath(step=step, batch_size=batch_size)
        is_update = os.path.exists(new_chkpt_path)
        # Save state_dict locally
        torch.save(state_dict, new_chkpt_path)
        # Upload new checkpoint file to Google Drive
        return self.upload_checkpoint(chkpt_filename=new_chkpt_path, batch_size=batch_size, delete_after=delete_after,
                                      in_parallel=in_parallel, show_progress=show_progress, is_update=is_update)

    def upload_checkpoint(self, chkpt_filename: str, batch_size: Optional[int] = None, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, GDriveFile or ColabFile or None]:
        # Check if needed to create the folder in Google Drive before uploading
        if batch_size is None:
            batch_size = -1
        # Ensure folder exists locally & in Google Drive
        self.ensure_chkpts_gfolder_exists(batch_size=batch_size)
        # Upload local file to Google Drive
        upload_gfolder = self.chkpts_batch_gfolders[batch_size]
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
        # Search for the checkpoint in the list with all model checkpoints inside checkpoints folder for given batch
        local_filepath = f'{self.configurations_gfolder.local_root}/{config_id_str}.yaml'
        return local_filepath if os.path.exists(local_filepath) and os.path.isfile(local_filepath) \
            else False

    def list_all_configurations(self, only_keys: Optional[Sequence] = None) -> List[GDriveFile or ColabFile or dict]:
        # Get checkpoint files list and filter it if only_keys attribute is set
        config_files_list = self.configurations_gfolder.files
        return config_files_list if not only_keys else \
            [dict((k, _f[k]) for k in only_keys) for _f in config_files_list]

    def save_and_upload_configuration(self, configuration: dict, config_id: Optional[str or int] = None,
                                      delete_after: bool = False, in_parallel: bool = False,
                                      show_progress: bool = False) -> ApplyResult or GDriveFile or ColabFile or None:
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
