import abc
from contextlib import contextmanager
from multiprocessing.pool import ApplyResult
from typing import Any, Union, List, Optional, Dict, Sequence, Type

from PIL.Image import Image
from torch import Tensor, nn


class _IFaceTemplate(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def _abstract_method(self) -> Any:
        """
        Example of an abstract method that should be implemented by classes that implement (or inherit since Python does
        not support interfaces) this interface class.
        :return: Any
        """
        raise NotImplementedError


class FilesystemCapsule(metaclass=abc.ABCMeta):
    """
    FilesystemCapsule Interface:
    The classes that implement `FilesystemCapsule` should be used as an open channel to "talk" to cloud storage
    services.
    """

    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'


class FilesystemFile(dict, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def download(self, in_parallel: bool = False, show_progress: bool = False, unzip_after: bool = False) \
            -> Union[ApplyResult, bool]:
        """
        Download the file from cloud storage to local filesystem at pre-specified :attr:`self.local_filepath`
        maintaining cloud file's basename.
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :param (bool) unzip_after: set to True to unzip the downloaded file when download completes successfully using
                                   the `unzip` lib & shell command
        :return: a `multiprocessing.pool.ApplyResult` object if :attr:`in_parallel` was set, else a bool object set to
                 True if download completed successfully, False with corresponding messages otherwise
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def folder(self) -> 'FilesystemFolder':
        """
        Get the parent `utils.ifaces.FilesystemFolder` instance of this cloud file instance. This is the folder inside
        of which the files in cloud as well as in local storage exist.
        :return: an `utils.ifaces.FilesystemFolder` instance or `None` with corresponding messages if errors occurred
        """
        raise NotImplementedError

    @folder.setter
    @abc.abstractmethod
    def folder(self, f: 'FilesystemFolder') -> None:
        """
        Set the (parent) folder instance of this cloud file instance.
        :param f: an `utils.ifaces.FilesystemFolder` instance
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_downloaded(self) -> bool:
        """
        Check whether the cloud file is downloaded and is present in the local filesystem.
        :return: an `bool` object
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the file name of this instance. This must be the basename of the local file AND the file name in
        cloud storage.
        :return: a `str` object with the file name
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path(self) -> str:
        """
        Get the absolute local path of this file instance whether or not this file has been downloaded locally.
        :return: a `str` object with the absolute local filepath
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
       Get the file size of this file instance in bytes.
       :return: a `int` object with the file size in bytes
       """
        raise NotImplementedError


class FilesystemFolder(dict, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def local_root(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def create_subfolder(self, folder_name: str, force_create_local: bool = False) -> 'FilesystemFolder':
        """
        Create a subfolder with the given :attr:`folder_name` in cloud storage.
        :param (str) folder_name: the name of the subfolder to be created
        :param (bool) force_create_local: set to True to create local folder immediately after creating folder in cloud
                                          storage; the local folder is created if not exists before any file upload and
                                          download
        :return: a `utils.ifaces.FilesystemFolder` instance to interact with the newly created subfolder
        """
        raise NotImplementedError

    @abc.abstractmethod
    def download(self, recursive: bool = False, in_parallel: bool = False, show_progress: bool = False,
                 unzip_after: bool = False) -> List[Union[ApplyResult, bool]]:
        """
        Downloads all the files inside the folder from cloud storage to local filesystem.
        :param (bool) recursive: set to True to download files inside subfolders in a recursive manner
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :param (bool) unzip_after: set to True to unzip all the zip downloaded files using the `unzip` lib
        :return: a list of `multiprocessing.pool.ApplyResult` objects if :attr:`in_parallel` was set, else a list of
                 `bool` objects with each set to True if the corresponding file download completed successfully,
                 `False`, with error messages displayed, otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def download_file(self, filename_or_cloud_file: Union[str, FilesystemFile], in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        """
        Downloads the file named after :attr:`filename` inside this folder instance, from cloud storage to local
        filesystem.
        :param (str or FilesystemFile) filename_or_cloud_file: the basename of the file to be downloaded as a string or
                                                               a  `utils.ifaces.FilesystemFile` instance
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :param (bool) unzip_after: set to True to unzip the downloaded file using the `unzip` lib & shell command
        :return: a `multiprocessing.pool.ApplyResult` object if :attr:`in_parallel` was set, else a bool object set to
                 True if download file completed successfully, False with corresponding messages otherwise
        :raises FileNotFoundError: if no file with given :attr:`filename` was found inside cloud folder root
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_local_root_exists(self):
        """
        Checks if local directory exists and if not creates it using the shell command `mkdir -p {self.local_root}`
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Get the folder name of this instance. This must be the basename of the local folder root AND the folder name in
        cloud drive storage.
        :return: an `str` object with the folder name
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def files(self) -> List[FilesystemFile]:
        """
        List all cloud files under the pre-specified folder described in :attr:`self.cloud_root`.
        :return: a `list` of `utils.ifaces.FilesystemFile` objects, each containing cloud file info of files found
                 inside the pre-specified cloud folder
        """
        raise NotImplementedError

    @abc.abstractmethod
    def file_by_name(self, filename: str) -> Optional[FilesystemFile]:
        """
        Get the cloud file instance that corresponds to a file inside this folder instance and whose file name
        matches the given :attr:`filename`.
        :param filename: the basename of the file to be searched for as a string
        :return: a `utils.ifaces.FilesystemFile` instance or None if no file found with given name or errors occurred
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parent(self) -> Optional['FilesystemFolder']:
        """
        Get the parent `utils.ifaces.FilesystemFolder` instance of this cloud folder instance.
        :return: an `utils.ifaces.FilesystemFolder` instance or `None` if cloud folder is under root or errors occurred
        """
        raise NotImplementedError

    @parent.setter
    @abc.abstractmethod
    def parent(self, p: Optional['FilesystemFolder']) -> None:
        """
        Set the parent instance of this cloud folder instance or `None` if folder is under root.
        :param (optional) p: an `utils.ifaces.FilesystemFolder` instance or `None` if cloud folder is under root
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subfolders(self) -> List['FilesystemFolder']:
        """
        Get all cloud folders under the pre-specified folder described in :attr:`self.cloud_root`.
        :return: a `list` of `utils.ifaces.FilesystemFolder` objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def subfolder_by_name(self, folder_name: str, recursive: bool = False) -> Optional['FilesystemFolder']:
        """
        Get the `utils.ifaces.FilesystemFolder` instance that corresponds to a subfolder of this instance matching the
        given.
        :attr:`folder_name`.
        :param (str) folder_name: the name of the subfolder to retrieve
        :param (bool) recursive: set to True to search inside subfolders of subfolders in a recursive manner to find the
                                 folder with the given :attr:`folder_name`
        :return: a `utils.ifaces.FilesystemFolder` object or None with corresponding messages if error(s) occurred
        """
        raise NotImplementedError

    @abc.abstractmethod
    def subfolder_by_name_or_create(self, folder_name: str, recursive: bool = False) -> Optional['FilesystemFolder']:
        """
        Get the `utils.ifaces.FilesystemFolder` instance that corresponds to a subfolder of this instance matching the
        given :attr:`folder_name`. If no subfolder matching the folder name found, `self.create_subfolder` is called to
        create a new subfolder in cloud as well as in local filesystem.
        :param (str) folder_name: the name of the subfolder to retrieve
        :param (bool) recursive: set to True to search inside subfolders of subfolders in a recursive manner to find the
                                 folder with the given :attr:`folder_name`
        :return: a `utils.ifaces.FilesystemFolder` object or None with corresponding messages if error(s) occurred
        """
        raise NotImplementedError

    @abc.abstractmethod
    def upload_file(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
                    show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, Optional[FilesystemFile]]:
        """
        Upload a locally-saved file to cloud drive at pre-specified cloud folder described by :attr:`self.cloud_root`.
        :param (str) local_filename: the basename of the local file (should exist inside :attr:`self.local_root`)
        :param (bool) delete_after: set to True to have the local file deleted after successful upload to cloud
        :param (bool) in_parallel: set to True to run the upload method in a separate thread, thus returning immediately
                                   to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the uploading progress printed using the `tqdm` lib
        :param (bool) is_update: set to True to update file in cloud storage instead of inserting a new one (if exists)
        :return: a `multiprocessing.pool.ApplyResult` object if `in_parallel` was set else a
                 `utils.ifaces.FilesystemFile` object with the uploaded cloud file info or `None` in case of failure
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def root(capsule_or_fs: Union[FilesystemCapsule, 'Filesystem']) -> 'FilesystemFolder':
        """
        Get the a `utils.ifaces.FilesystemFolder` instance to interact with Google Drive's root folder.
        :param (FilesystemCapsule or Filesystem) capsule_or_fs: a `utils.ifaces.Filesystem` instance or a
                                                                `utils.ifaces.FilesystemCapsule` instance to create the
                                                                filesystem instance and consequently crate the
                                                                `utils.ifaces.FilesystemFolder` instance
        :return: a new `utils.filesystems.local.FilesystemFolder` instance to interact with Google Drive's root folder
                 or the one WE consider as root
        """
        raise NotImplementedError


class Filesystem(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    def create_folder(self, cloud_folder: FilesystemFolder, folder_name: str, force_create_local: bool = False) \
            -> Optional[FilesystemFolder]:
        """
        Create a new folder under given :attr:`cloud_folder` named after the given :attr:`folder_name`.
        :param (FilesystemFolder) cloud_folder: a `utils.ifaces.FilesystemFolder` instance with the folder inside which
                                                the new folder will be created in cloud storage
        :param (str) folder_name: the name of the folder to be created as an `str` object
        :param (bool) force_create_local: set to True to create local folder immediately after creating folder in cloud
                                          storage; the local folder is created if not exists before any file upload and
                                          download
        :return: a `utils.ifaces.FilesystemFolder` instance to interact with the newly created folder in cloud storage
                 or None with corresponding messages if errors occurred
        """
        raise NotImplementedError

    @staticmethod
    def download_file_thread(_self: 'Filesystem', cloud_file: FilesystemFile, local_filepath: str,
                             show_progress: bool = False, unzip_after: bool = False) -> bool:
        return _self.download_file(cloud_file, local_filepath=local_filepath, in_parallel=False,
                                   show_progress=show_progress, unzip_after=unzip_after)

    @abc.abstractmethod
    def download_file(self, cloud_file: FilesystemFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        """
        Download cloud file described in :attr:`chkpt_data` from cloud drive to local filesystem at the given
        :attr:`local_filepath`.
        :param (FilesystemFile) cloud_file: a `utils.ifaces.FilesystemFile` object with cloud file details
                                       (e.g. cloud file id, title, etc.)
        :param (str) local_filepath: the absolute path of the local file that the cloud file will be downloaded to
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :param (bool) unzip_after: set to True to unzip the downloaded file when download completes successfully using
                                   the `unzip` lib & shell command
        :return: a Thread object if in_parallel was set else a bool object set to True if upload completed successfully,
                 False with corresponding messages otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_files(self, cloud_folder: dict) -> List[FilesystemFile]:
        """
        List all cloud files under the folder described in :attr:`cloud_folder`.
        :param (dict) cloud_folder: a `dict`-like object containing cloud folder info (e.g. folder id, title, etc.)
        :return: a `list` of `utils.ifaces.FilesystemFile` objects, each containing cloud file info
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_folders(self, cloud_folder: dict) -> List[FilesystemFolder]:
        """
        List all cloud folders under the folder described in :attr:`cloud_folder`.
        :param (dict) cloud_folder: a `dict`-like object containing cloud folder info (e.g. folder id, title, etc.)
        :return: a `list` of `utils.ifaces.FilesystemFolder` objects, each containing cloud folder info
        """
        raise NotImplementedError

    @staticmethod
    def upload_local_file_thread(_self: 'Filesystem', local_filepath: str, cloud_folder: FilesystemFolder,
                                 delete_after: bool = False, show_progress: bool = False) -> FilesystemFile or None:
        return _self.upload_local_file(local_filepath=local_filepath, cloud_folder=cloud_folder,
                                       delete_after=delete_after, in_parallel=False, show_progress=show_progress)

    @abc.abstractmethod
    def upload_local_file(self, local_filepath: str, cloud_folder: dict, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, FilesystemFile or None]:
        """
        Upload locally-saved file from :attr:`local_filepath` to cloud drive folder described by
        :attr:`cloud_folder`.
        :param (str) local_filepath: the absolute path of locally-saved file to be uploaded to the cloud
        :param (dict) cloud_folder: a dict containing info to access the destination cloud folder
                                    (e.g. folder id, title, etc.)
        :param (bool) delete_after: set to True to have the local file deleted after successful upload to cloud
        :param (bool) in_parallel: set to True to run the upload method in a separate thread, thus returning immediately
                                   to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the uploading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if `in_parallel` was set, else a
                 `utils.ifaces.FilesystemFile` object with the uploaded cloud file info or `None` in case of failure
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def folder_cls() -> Type[FilesystemFolder]:
        """
        Get the folder class that this filesystem interacts with (e.g. for using static methods of that class).
        :return: a class name, for a class that implements the `utils.ifaces.FilesystemFolder` interface.
        """
        raise NotImplementedError


class FilesystemDataset(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def fetch_and_unzip(self, in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        """
        Fetch (download) main .zip file from cloud storage to local filesystem and the unzip it locally.
        :param (bool) in_parallel: set to True to run the download method (plus the unzipping) in a separate thread,
                                   thus returning immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if :attr:`in_parallel` was set, else a bool object set to
                 True if upload completed successfully, False with corresponding messages otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_fetched_and_unzipped(self) -> bool:
        """
        Check that dataset's main .zip file has been unzipped in local dataset root, at :attr:`self.local_root`. In
        particular, if for example the main .zip file is named `Img.zip` this method will return true if a `Img` folder
        is found inside in local dataset root, at :attr:`self.local_root`.
        :return: a `bool` object
        """
        raise NotImplementedError


class FilesystemModel(metaclass=abc.ABCMeta):
    latest_metrics = None
    visualizations_fs_folder = None

    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    #
    # ------------------------------------
    #  Model Checkpoints
    # ---------------------------------
    #
    # Below, are the methods to capture, save, upload and download model checkpoints from/to cloud storage.
    #

    @abc.abstractmethod
    def download_checkpoint(self, epoch_or_id: Union[int, str], step: Optional[int] = None,
                            in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        """
        Download model checkpoint at given epoch and step or with given id (e.g. for torchvision pretrained models)
        :param (int or str) epoch_or_id: this is used to define the folder inside which the wanted checkpoint lives
                                         or the checkpoint filename (without the extension) if is `str`
        :param (optional) step: if no step is specified it will return the latest found checkpoint at the folder
                                containing the model checkpoints of the given :attr:`epoch`. This is mutually exclusive
                                with a `str` provided for :attr:`epoch_or_id`
        :param (bool) in_parallel:
        :param (bool) show_progress:
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`step` and
                                   :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_checkpoint(self, epoch_or_id: Union[int, str], step: Optional[int] = None) -> str or False:
        """
        Checks if model checkpoint at given epoch and step or with given id has been downloaded else it downloads file
        locally and returns the absolute filepath.
        :param (int or str) epoch_or_id:
        :param (optional) step:
        :return: a `str` containing the fetched model checkpoint matching the given :attr:`epoch` and :attr:`step` or
                 the given :attr:`id`; or `False` with corresponding messages if no match found
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`epoch_or_id` and
                                   :attr:`step`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_latest_checkpoint(self, epoch: Optional[int] = None) -> str or False:
        """
        Downloads the latest model checkpoint found inside the epoch folder named after "epoch=<:attr:`epoch`>".
        :param (optional) epoch: if not set, then will fetch the latest model checkpoint from the folder with the
                          highest epoch number (epoch folders should be named after "epoch=<epoch>" in storage)
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_checkpoint_fetched(self, epoch_or_id: Union[int, str], step: Optional[int] = None) -> str or False:
        """
        Checks if model checkpoint is fetched and, if so, it returns the absolute file path of the local file.
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::fetch_checkpoint()`
        :param (optional) step: see `utils.ifaces.FilesystemModel::fetch_checkpoint()`
        :return: a `str` with the absolute local filepath if the model checkpoint has been downloaded, `False` otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_checkpoints(self, epoch: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[FilesystemFile or dict]:
        """
        Lists all model checkpoints inside the epoch folder named after "epoch=<:attr:`epoch`>".
        :param (optional) epoch: if not set, then will list the latest model checkpoint from the folder with the
                          highest epoch number
        :param (optional) only_keys: if set instead the entire file info dict for each found file, it will return just
                                     the provided keys
        :return: a `list` of `dict` objects if :attr:`only_keys` is set else a `list` of `utils.ifaces.FilesystemFile`
                 objects of the found model checkpoints for given :attr:`epoch` otherwise
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_all_checkpoints(self, only_keys: Optional[Sequence[str]] = None) \
            -> Dict[int, List[FilesystemFile or dict]]:
        """
        Lists all model checkpoints under "Checkpoints" folder as a {epoch: [epoch checkpoints]} dict.
        :param (optional) only_keys: if set instead the entire file info dict for each found file, it will return just
                                     the provided keys
        :return: a `list` of `dict` objects if :attr:`only_keys` is set else a `list` of `utils.ifaces.FilesystemFile`
                 objects of the found model checkpoints otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_and_upload_checkpoint(self, state_dict: dict, epoch_or_id: Union[int, str], step: Optional[int] = None,
                                   metrics_dict: Optional[dict] = None, delete_after: bool = False,
                                   in_parallel: bool = False, show_progress: bool = False) \
            -> Union[List[ApplyResult], List[FilesystemFile or None]]:
        """
        Save the given :attr:`state_dict` locally and then upload the saved checkpoint to cloud for permanent storage.
        :param (dict) state_dict: the model state dict (e.g. the result of calling `model.state_dict()`)
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (int or str) step: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (optional) metrics_dict: if provided will also save and upload a metrics (.json) file in the "Metrics"
                                        folder
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :param (bool) show_progress: set to True to have the uploading progress displayed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.ifaces.FilesystemFile` object if upload completed successfully, `None` with corresponding
                 messages otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def upload_checkpoint(self, chkpt_filename: str, epoch_or_id: Union[int, str], delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False, is_update: bool = False) \
            -> ApplyResult or FilesystemFile or None:
        """
        Upload locally-saved model checkpoint to cloud storage for permanent storage.
        :param (str) chkpt_filename: file name (NOT absolute path) of the locally-saved checkpoint file.
                                     The filename MUST follow the following naming convention:
                                     "<model_name: str>_<step: int or str>_<batch_size: Optional[int]>"
        :param (int or str) epoch_or_id: see `utils.ifaces.FilesystemModel::download_checkpoint()`
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :param (bool) show_progress: set to True to have the uploading progress displayed using the `tqdm` lib
        :param (bool) is_update: set to True to update file in cloud storage, else a new file will be inserted
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.ifaces.FilesystemFile` object if upload completed successfully, `None` with corresponding
                 messages otherwise
        """
        raise NotImplementedError

    #
    # ------------------------------------
    #  Model Metrics
    # ---------------------------------
    #
    # Below, are the methods to capture, save, upload and download model metrics from/to cloud storage.
    #

    @abc.abstractmethod
    def list_metrics(self, epoch: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[FilesystemFile or dict]:
        """
        Same as `utils.ifaces.FilesystemModel::list_checkpoints()` but the "Metrics" cloud folder.
        :param epoch: see `utils.ifaces.FilesystemModel::list_checkpoints()`
        :param only_keys: see `utils.ifaces.FilesystemModel::list_checkpoints()`
        :return: see `utils.ifaces.FilesystemModel::list_checkpoints()`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_all_metrics(self, only_keys: Optional[Sequence[str]] = None) -> Dict[int, List[FilesystemFile or dict]]:
        """
        Lists all model metric files under "Metrics" folder as a {epoch: [epoch checkpoints]} dict.
        :param (optional) only_keys: if set instead the entire file info dict for each found file, it will return just
                                     the provided keys
        :return: a `list` of `dict` objects if :attr:`only_keys` is set else a `list` of `utils.ifaces.FilesystemFile`
                 objects of the found model metrics otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_metrics(self, epoch: Optional[int] = None) -> List[FilesystemFile]:
        """
        Re-run evaluator for checkpoints of the given :att:`epoch`, updating existing metrics.
        :param epoch: see `utils.ifaces.FilesystemModel::list_checkpoints()`
        :return: see `utils.ifaces.FilesystemModel::list_checkpoints()`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_all_metrics(self) -> Dict[int, List[FilesystemFile]]:
        """
        Re-run evaluator for all model checkpoints, updating all existing metrics.
        :return: a `list` of `utils.ifaces.FilesystemFile` objects of the updated model metrics
        """
        raise NotImplementedError

    #
    # ------------------------------------
    #  Model Configurations
    # ---------------------------------
    #
    # Below, are the methods to capture, save, upload and download model configurations to/from cloud storage.
    #

    @abc.abstractmethod
    def download_configuration(self, config_id: Union[int, str], in_parallel: bool = False,
                               show_progress: bool = False) -> Union[ApplyResult, bool]:
        """TODO fill documentation"""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_configuration(self, config_id: Union[int, str]) -> str or False:
        """TODO fill documentation"""
        raise NotImplementedError

    @abc.abstractmethod
    def is_configuration_fetched(self, config_id: Union[int, str]) -> str or False:
        """TODO fill documentation"""
        raise NotImplementedError

    @abc.abstractmethod
    def list_configurations(self, only_keys: Optional[Sequence] = None) -> List[FilesystemFile or dict]:
        """TODO fill documentation"""
        raise NotImplementedError

    @abc.abstractmethod
    def save_and_upload_configuration(self, config: dict, config_id: Optional[str or int] = None,
                                      delete_after: bool = False, in_parallel: bool = False,
                                      show_progress: bool = False) -> Union[ApplyResult, FilesystemFile or None]:
        """TODO fill documentation"""
        raise NotImplementedError

    @abc.abstractmethod
    def upload_configuration(self, config_filename: str, delete_after: bool = False, in_parallel: bool = False,
                             show_progress: bool = False) -> Union[ApplyResult, FilesystemFile or None]:
        """TODO fill documentation"""
        raise NotImplementedError


class Configurable(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def load_configuration(self, configuration: dict) -> None:
        """
        Load given :attr:`configuration` (that resulted from yaml.load()) to the model instance.
        :param (dict) configuration: the model configuration to be loaded
        """
        raise NotImplementedError

    @abc.abstractmethod
    def configuration(self) -> dict:
        """
        Get current model configuration as a `dict` object.
        :return: a `dict` object with the current model configuration
        """
        raise NotImplementedError


class Evaluable(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def evaluate(self, metric_name: Optional[str] = None, show_progress: bool = True) \
            -> Union[Dict[str, Tensor or float], Tensor or float]:
        """
        Evaluate current model's state and return a `dict` with metric names as keys and evaluation results as values.
        :param (optional) metric_name: the name of the evaluation metric to be applied
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :return: if :attr:`metric` is `None` then a `dict` of all available metrics is returned, only the given metric
                 is returned otherwise
        """
        raise NotImplementedError


class Freezable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def freeze(self, force: bool = False) -> None:
        """
        Freezes model (sets requires_grad=False to all learnable parameters)
        :param (bool) force: set to True any intermediate checks before freezing the network.
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self, force: bool = False) -> None:
        """
        Unfreezes model (sets requires_grad=True to all learnable parameters)
        :param (bool) force: set to True any intermediate checks before freezing the network.
        :return: None
        """
        raise NotImplementedError

    @contextmanager
    def frozen(self) -> 'Freezable':
        """
        Manages context by freezing model, yielding it and then unfreezing it
        :return: self instance
        """
        self.freeze()
        yield self
        self.unfreeze()


class BalancedFreezable(Freezable):
    """
    BalancedFreezable Class:
    To be used in cases where nested calls to `with module.frozen()` may arise. Does not unfreezes the model until the
    number of `unfreeze()` calls reaches the corresponding number of `freeze()` calls.
    """

    def __init__(self):
        """
        BalancedFreezable class constructor.
        """
        self._freeze_requests_count: int = 0
        self._state = 'unfrozen'

    def reset_freeze_state(self) -> None:
        self.unfreeze(force=True)
        self._freeze_requests_count = 0

    def freeze(self, force: bool = False) -> None:
        assert isinstance(self, nn.Module), 'self must be a nn.Module to apply freezing'
        if force:
            for p in self.parameters():
                p.requires_grad = False
        else:
            self._freeze_requests_count += 1
            if self._freeze_requests_count == 1:
                for p in self.parameters():
                    p.requires_grad = False
                self._state = 'frozen'

    def unfreeze(self, force: bool = False) -> None:
        assert isinstance(self, nn.Module), 'self must be a nn.Module to apply freezing'
        if force:
            for p in self.parameters():
                p.requires_grad = True
        else:
            self._freeze_requests_count -= 1
            if self._freeze_requests_count == 0:
                assert isinstance(self, nn.Module), 'self must be a nn.Module to apply unfreezing'
                for p in self.parameters():
                    p.requires_grad = True
                self._state = 'unfrozen'
            assert self._freeze_requests_count >= 0, f'self.freeze_requests_count={self._freeze_requests_count}'


class Reproducible(metaclass=abc.ABCMeta):
    _seed = None

    @staticmethod
    @abc.abstractmethod
    def manual_seed(seed: int) -> None:
        """
        Achieve reproducibility using a manual seeder.
        :param (int) seed: seeder value
        :return: None
        """
        raise NotImplementedError

    @staticmethod
    def is_seeded() -> bool:
        """
        Check if manual_seed() has been called before.
        :return: True if _seed has been set else False
        """
        return Reproducible._seed is not None


class ResumableDataLoader(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def get_state(self) -> dict:
        """
        Return dataloader current state (e.g. current indices list and current index)
        :return: an dict object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state: dict) -> None:
        """
        Set dataloader current state (e.g. current indices list and current index)
        :param (dict) state: a dict with same structure as the one returned by `ResumableDataLoader::get_state()`
        """
        raise NotImplementedError


class Verbosable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_layer_attr_names(self) -> List[str]:
        """
        Get a list of all module attributes that are themselves nn.Module instances.
        :return: a list object of attribute names
        """
        raise NotImplementedError


class Visualizable(metaclass=abc.ABCMeta):
    """
    Visualizable Abstract Class (aka Interface)
    """

    def __init__(self, indices: Sequence = (0, -1)):
        """
        Visualizable abstract class constructor.
        :param (Sequence) indices:
        """
        self._reproducible_indices = indices

    @classmethod
    def version(cls) -> str:
        """
        Get interface version
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @property
    def reproducible_indices(self) -> Sequence:
        """
        Get indices list to visualize reproducible images.
        :return: a tuple object (defaults to (0, -1))
        """
        return self._reproducible_indices

    @reproducible_indices.setter
    def reproducible_indices(self, indices: Sequence) -> None:
        """
        Set indices list to visualize reproducible images.
        :param (Sequence) indices: a tuple object (defaults to (0, -1))
        """
        self._reproducible_indices = indices

    @abc.abstractmethod
    def visualize(self, reproducible: bool = False) -> Image:
        """
        Visualize latest model's forward pass by creating a `PIL.Image.Image` object with the model output and possible
        its forward pass's inputs.
        :param (bool) reproducible: set to True to have the visualizer echo the same images every time the
                                    visualize() method is called
        :return: a `PIL.Image.Image` object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def visualize_indices(self, indices: tuple) -> Image:
        """
        Visualize images generated by fetching dataset's images at specified index.
        :param (tuple) indices: dataset's indices to fetch as source / target images
        :return: a `PIL.Image.Image` object
        """

    @abc.abstractmethod
    def visualize_metrics(self, upload: bool = False, preview: bool = False) -> List[Image]:
        """
        Aggregate all metrics found in model's "Metrics" folder and present them in pretty plots. Class MUST also
        implement `utils.ifaces.FilesystemModel` for this method to work.
        :param (bool) upload: set to True to have produced plots uploaded to cloud storage
        :param (bool) preview: set to True to have the produced plots displayed inline using `plt.show()`
        :return: a `list` of `PIL.Image.Image` objects containing the visualizations for the respective metrics
        """
        raise NotImplementedError
