import abc
from multiprocessing.pool import ApplyResult
from typing import Any, Union, List, Optional

from pydrive.files import GoogleDriveFile


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
    The classes that implement `CloudCapsule` should be used as an open channel to "talk" to cloud services.
    """

    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'


class CloudFilesystem(metaclass=abc.ABCMeta):
    @classmethod
    def version(cls) -> str:
        """
        Get interface version.
        :return: a string with the current interface version (e.g. 1.0)
        """
        return '1.0'

    @staticmethod
    def download_file_thread(_self: 'CloudFilesystem', cloud_file: GoogleDriveFile, local_filepath: str) -> bool:
        return _self.download_file(cloud_file=cloud_file, local_filepath=local_filepath,
                                   in_parallel=False, show_progress=False)

    @abc.abstractmethod
    def download_file(self, cloud_file: GoogleDriveFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False) -> Union[ApplyResult, bool]:
        """
        Download cloud file described in :attr:`chkpt_data` from cloud drive to local filesystem at the given
        :attr:`local_filepath`.
        :param (GoogleDriveFile) cloud_file: a `pydrive.files.GoogleDriveFile` object with cloud file details
                                             (e.g. cloud file id, title, etc.)
        :param (str) local_filepath: the absolute path of the local file that the cloud file will be downloaded to
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :return: a Thread object if in_parallel was set else a bool object set to True if upload completed successfully,
                 False with corresponding messages otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_files(self, cloud_folder: GoogleDriveFile) -> List[GoogleDriveFile]:
        """
        List all cloud files under the folder described in :attr:`cloud_folder`.
        :param (GoogleDriveFile) cloud_folder: a `pydrive.files.GoogleDriveFile` object containing cloud folder info
                                               (e.g. folder id, title, etc.)
        :return: a `list` of `pydrive.files.GoogleDriveFile` objects, each containing cloud file info
        """
        raise NotImplementedError

    @staticmethod
    def upload_local_file_thread(_self: 'CloudFilesystem', cloud_folder: GoogleDriveFile, local_filepath: str,
                                 delete_after: bool = False) -> Optional[GoogleDriveFile]:
        return _self.upload_local_file(local_filepath=local_filepath, cloud_folder=cloud_folder,
                                       delete_after=delete_after, in_parallel=False, show_progress=False)

    @abc.abstractmethod
    def upload_local_file(self, local_filepath: str, cloud_folder: GoogleDriveFile, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, Optional[GoogleDriveFile]]:
        """
        Upload locally-saved file from :attr:`local_filepath` to cloud drive folder described by
        :attr:`cloud_folder`.
        :param (str) local_filepath: the absolute path of locally-saved file to be uploaded to the cloud
        :param (GoogleDriveFile) cloud_folder: a dict containing info to access the destination cloud folder
                                               (e.g. folder id, title, etc.)
        :param (bool) delete_after: set to True to have the local file deleted after successful upload to cloud
        :param (bool) in_parallel: set to True to run the upload method in a separate thread, thus returning immediately
                                   to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the uploading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if `in_parallel` was set, else a
                 `pydrive.files.GoogleDriveFile` object with the uploaded cloud file info or None in case of failure
        """
        raise NotImplementedError


class CloudFolder(dict, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def download(self, cloud_file: GoogleDriveFile, in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, bool]:
        """
        Download file from cloud drive described by :attr:`chkpt_data` to local filesystem at pre-specified
        :attr:`self.local_root` maintaining cloud drive's file basename.
        :param (GoogleDriveFile) cloud_file: a `pydrive.files.GoogleDriveFile` object with cloud file details
                                             (e.g. cloud file id, title, etc.)
        :param (bool) in_parallel: set to True to run the download method in a separate thread, thus returning
                                   immediately to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the downloading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if :attr:`in_parallel` was set, else a bool object set to
                 True if upload completed successfully, False with corresponding messages otherwise
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list(self) -> List[GoogleDriveFile]:
        """
        List all cloud files under the pre-specified folder described in :attr:`self.cloud_root`.
        :return: a `list` of `pydrive.files.GoogleDriveFile` objects, each containing cloud file info of files found
                 inside the pre-specified cloud folder
        """
        raise NotImplementedError

    @abc.abstractmethod
    def upload(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
               show_progress: bool = False) -> Union[ApplyResult, Optional[GoogleDriveFile]]:
        """
        Upload locally-saved file to cloud drive at pre-specified cloud folder described by :attr:`self.cloud_root`.
        :param (str) local_filename: the basename of the local file (should exist inside :attr:`local_root_path`)
        :param (bool) delete_after: set to True to have the local file deleted after successful upload to cloud
        :param (bool) in_parallel: set to True to run the upload method in a separate thread, thus returning immediately
                                   to the caller with a thread-related object
        :param (bool) show_progress: set to True to have the uploading progress printed using the `tqdm` lib
        :return: a `multiprocessing.pool.ApplyResult` object if `in_parallel` was set else a
                 `pydrive.files.GoogleDriveFile` object with the uploaded cloud file info or None in case of failure
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def instance(fs: CloudFilesystem, cloud_root: GoogleDriveFile, local_root: str) -> 'CloudFolder':
        """
        Instantiate a `CloudFolder` instance to sync data between cloud and local folders.
        :param (CloudFilesystem) fs: a `CloudFilesystem` instance to interact with the cloud filesystem
        :param (GoogleDriveFile) cloud_root: a `pydrive.files.GoogleDriveFile` containing info to access the cloud
                                             folder (e.g. folder id, title, etc.)
        :param (str) local_root: the absolute path of the local folder that the cloud files will be downloaded to
        :return: a `CloudFolder` instance
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
