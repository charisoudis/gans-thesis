import abc
from multiprocessing.pool import ApplyResult
from typing import Any, Union, List, Optional, Dict, Sequence


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


class CloudCapsule(metaclass=abc.ABCMeta):
    """
    CloudCapsule Interface:
    The classes that implement `CloudCapsule` should be used as an open channel to "talk" to cloud storage services.
    """

    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'


class CloudFile(dict, metaclass=abc.ABCMeta):
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
        :return: an `str` object with the file name
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def folder(self) -> 'CloudFolder':
        """
        Get the parent `utils.ifaces.CloudFolder` instance of this cloud file instance. This is the folder inside of
        which lives the file in cloud as well as in local storage.
        :return: an `utils.ifaces.CloudFolder` instance or `None` with corresponding messages if errors occurred
        """
        raise NotImplementedError

    @folder.setter
    @abc.abstractmethod
    def folder(self, f: 'CloudFolder') -> None:
        """
        Set the (parent) folder instance of this cloud file instance.
        :param f: an `utils.ifaces.CloudFolder` instance
        """
        raise NotImplementedError


class CloudFolder(dict, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_subfolder(self, folder_name: str, force_create_local: bool = False) -> 'CloudFolder':
        """
        Create a subfolder with the given :attr:`folder_name` in cloud storage.
        :param (str) folder_name: the name of the subfolder to be created
        :param (bool) force_create_local: set to True to create local folder immediately after creating folder in cloud
                                          storage; the local folder is created if not exists before any file upload and
                                          download
        :return: a `utils.ifaces.CloudFolder` instance to interact with the newly created subfolder
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
    def download_file(self, filename_or_cloud_file: Union[str, CloudFile], in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        """
        Downloads the file named after :attr:`filename` inside this folder instance, from cloud storage to local
        filesystem.
        :param (str or CloudFile) filename_or_cloud_file: the basename of the file to be downloaded as a string or an
                                                          `utils.ifaces.CloudFile` instance
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :param (bool) unzip_after: set to True to unzip the downloaded file using the `unzip` lib & shell command
        :return: a `multiprocessing.pool.ApplyResult` object if :attr:`in_parallel` was set, else a bool object set to
                 True if download file completed successfully, False with corresponding messages otherwise
        :raises FileNotFoundError: if no file with given :attr:`filename` was found inside cloud folder root
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
    def files(self) -> List[CloudFile]:
        """
        List all cloud files under the pre-specified folder described in :attr:`self.cloud_root`.
        :return: a `list` of `utils.ifaces.CloudFile` objects, each containing cloud file info of files found
                 inside the pre-specified cloud folder
        """
        raise NotImplementedError

    @abc.abstractmethod
    def file_by_name(self, filename: str) -> Optional[CloudFile]:
        """
        Get the cloud file instance that corresponds to a file inside this folder instance and whose file name
        matches the given :attr:`filename`.
        :param filename: the basename of the file to be searched for as a string
        :return: a `utils.ifaces.CloudFile` instance or None if no file found with given name or errors occurred
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parent(self) -> Optional['CloudFolder']:
        """
        Get the parent `utils.ifaces.CloudFolder` instance of this cloud folder instance.
        :return: an `utils.ifaces.CloudFolder` instance or `None` if cloud folder is under root or errors occurred
        """
        raise NotImplementedError

    @parent.setter
    @abc.abstractmethod
    def parent(self, p: Optional['CloudFolder']) -> None:
        """
        Set the parent instance of this cloud folder instance or `None` if folder is under root.
        :param (optional) p: an `utils.ifaces.CloudFolder` instance or `None` if cloud folder is under root
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subfolders(self) -> List['CloudFolder']:
        """
        Get all cloud folders under the pre-specified folder described in :attr:`self.cloud_root`.
        :return: a `list` of `utils.ifaces.CloudFolder` objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def subfolder_by_name(self, folder_name: str, recursive: bool = False) -> Optional['CloudFolder']:
        """
        Get the `utils.ifaces.CloudFolder` instance that corresponds to a subfolder of this instance matching the given
        :attr:`folder_name`.
        :param (str) folder_name: the name of the subfolder to retrieve
        :param (bool) recursive: set to True to search inside subfolders of subfolders in a recursive manner to find the
                                 folder with the given :attr:`folder_name`
        :return: a `utils.ifaces.CloudFolder` object or None with corresponding messages if error(s) occurred
        """
        raise NotImplementedError

    @abc.abstractmethod
    def upload_file(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
                    show_progress: bool = False) -> Union[ApplyResult, Optional[CloudFile]]:
        """
        Upload a locally-saved file to cloud drive at pre-specified cloud folder described by :attr:`self.cloud_root`.
        :param (str) local_filename: the basename of the local file (should exist inside :attr:`self.local_root`)
        :param (bool) delete_after: set to True to have the local file deleted after successful upload to cloud
        :param (bool) in_parallel: set to True to run the upload method in a separate thread, thus returning immediately
                                   to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the uploading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if `in_parallel` was set else a
                 `utils.ifaces.CloudFile` object with the uploaded cloud file info or `None` in case of failure
        """
        raise NotImplementedError


class CloudFilesystem(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    def create_folder(self, cloud_folder: CloudFolder, folder_name: str, force_create_local: bool = False) \
            -> Optional[CloudFolder]:
        """
        Create a new folder under given :attr:`cloud_folder` named after the given :attr:`folder_name`.
        :param (CloudFolder) cloud_folder: a `utils.ifaces.CloudFolder` instance with the folder inside which the new
                                           folder will be created in cloud storage
        :param (str) folder_name: the name of the folder to be created as an `str` object
        :param (bool) force_create_local: set to True to create local folder immediately after creating folder in cloud
                                          storage; the local folder is created if not exists before any file upload and
                                          download
        :return: a `utils.ifaces.CloudFolder` instance to interact with the newly created folder in cloud storage or
                 None with corresponding messages if errors occurred
        """
        raise NotImplementedError

    @staticmethod
    def download_file_thread(_self: 'CloudFilesystem', cloud_file: CloudFile, local_filepath: str,
                             show_progress: bool = False, unzip_after: bool = False) -> bool:
        return _self.download_file(cloud_file, local_filepath=local_filepath, in_parallel=False,
                                   show_progress=show_progress, unzip_after=unzip_after)

    @abc.abstractmethod
    def download_file(self, cloud_file: CloudFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        """
        Download cloud file described in :attr:`chkpt_data` from cloud drive to local filesystem at the given
        :attr:`local_filepath`.
        :param (CloudFile) cloud_file: a `utils.ifaces.CloudFile` object with cloud file details
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
    def list_files(self, cloud_folder: dict) -> List[CloudFile]:
        """
        List all cloud files under the folder described in :attr:`cloud_folder`.
        :param (dict) cloud_folder: a `dict`-like object containing cloud folder info (e.g. folder id, title, etc.)
        :return: a `list` of `utils.ifaces.CloudFile` objects, each containing cloud file info
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_folders(self, cloud_folder: dict) -> List[CloudFolder]:
        """
        List all cloud folders under the folder described in :attr:`cloud_folder`.
        :param (dict) cloud_folder: a `dict`-like object containing cloud folder info (e.g. folder id, title, etc.)
        :return: a `list` of `utils.ifaces.CloudFolder` objects, each containing cloud folder info
        """
        raise NotImplementedError

    @staticmethod
    def upload_local_file_thread(_self: 'CloudFilesystem', local_filepath: str, cloud_folder: CloudFolder,
                                 delete_after: bool = False) -> CloudFile or None:
        return _self.upload_local_file(local_filepath=local_filepath, cloud_folder=cloud_folder,
                                       delete_after=delete_after, in_parallel=False, show_progress=False)

    @abc.abstractmethod
    def upload_local_file(self, local_filepath: str, cloud_folder: dict, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, CloudFile or None]:
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
                 `utils.ifaces.CloudFile` object with the uploaded cloud file info or `None` in case of failure
        """
        raise NotImplementedError


class CloudDataset(metaclass=abc.ABCMeta):
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


class CloudModel(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @abc.abstractmethod
    def download_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None,
                            in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        """
        TODO fill documentation
        :param step:
        :param batch_size:
        :param in_parallel:
        :param show_progress:
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`step` and
                                   :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
        """
        TODO
        :param step:
        :param batch_size:
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`step` and
                                   :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_latest_checkpoint(self, batch_size: Optional[int] = None) -> str or False:
        """
        TODO
        :param batch_size:
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_checkpoint_fetched(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
        """
        TODO fill documentation
        :param step:
        :param batch_size:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_checkpoints(self, batch_size: Optional[Sequence[str]] = None, only_keys: Optional[list] = None) \
            -> List[CloudFile or dict]:
        """
        TODO fill documentation
        :param batch_size:
        :param only_keys:
        :return:
        :raises FileNotFoundError: if no checkpoint file could be found matching the given :attr:`batch_size`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_all_checkpoints(self, only_keys: Optional[Sequence[str]] = None) -> Dict[int, List[CloudFile or dict]]:
        """
        TODO fill documentation
        :param only_keys:
        :return:
        """
        raise NotImplementedError

    def save_and_upload_checkpoint(self, state_dict: dict, step: Union[int, str], batch_size: Optional[int] = None,
                                   delete_after: bool = False, in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, CloudFile or None]:
        """
        Save the given :attr:`state_dict` locally and then upload the saved checkpoint to cloud for permanent storage.
        :param (dict) state_dict: the model state dict (e.g. the result of calling `model.state_dict()`)
        :param (int or str) step: see `utils.ifaces.CloudModel::download_checkpoint`
        :param (optional) batch_size: see `utils.ifaces.CloudModel::download_checkpoint`
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :param (bool) show_progress: set to True to have the uploading progress displayed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.ifaces.CloudFile` object if upload completed successfully, `None` with corresponding messages
                 otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def upload_checkpoint(self, chkpt_filename: str, batch_size: Optional[int] = None, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, CloudFile or None]:
        """
        Upload locally-saved model checkpoint to cloud storage for permanent storage.
        :param (str) chkpt_filename: file name (NOT absolute path) of the locally-saved checkpoint file.
                                     The filename MUST follow the following naming convention:
                                     "<model_name: str>_<step: int or str>_<batch_size: Optional[int]>"
        :param (optional) batch_size: see `utils.ifaces.CloudModel::download_checkpoint`
        :param (bool) delete_after: set to True to have the local file deleted after successful upload
        :param (bool) in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                                   to caller
        :param (bool) show_progress: set to True to have the uploading progress displayed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else an
                 `utils.ifaces.CloudFile` object if upload completed successfully, `None` with corresponding messages
                 otherwise
        """
        raise NotImplementedError


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
