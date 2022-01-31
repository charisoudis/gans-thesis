import atexit
import os
import zipfile
from multiprocessing.pool import ApplyResult, ThreadPool
from typing import Optional, List, Union, Type

from utils.command_line_logger import CommandLineLogger
from utils.data import unzip_file
from utils.ifaces import FilesystemCapsule, Filesystem, FilesystemFile, FilesystemFolder


class LocalCapsule(FilesystemCapsule):
    """
    LocalCapsule Class:
    This is class is used for API compatibility (implementing `utils.ifaces.FilesystemCapsule`)
    """

    def __init__(self, local_root: str):
        """
        LocalCapsule class constructor.
        :param (str) local_root: absolute path to the local directory where local files will be placed in
        """
        self.local_root = local_root
        self.logger = CommandLineLogger(log_level=os.getenv('TRAIN_LOG_LEVEL', 'info'), name=self.__class__.__name__)


class LocalFile(FilesystemFile):
    """
    LocalFile Class:
    This class, implementing `utils.ifaces.FilesystemFile` interface, is used to interact with files stored in the
    local filesystem.
    """

    def __init__(self, filename: str, cfolder: 'LocalFolder'):
        """
        LocalFile class constructor.
        :param (str) filename: basename of file in local fs
        :param (LocalFolder) cfolder: an `utils.filesystems.local.LocalFolder` instance with the folder info inside of
                                      which lives this local file
        """
        # Save args
        self.filename = filename
        self.cfolder = cfolder
        # Form local file path (absolute)
        self.filepath = f'{self.cfolder.local_root}/{filename}'
        self.filesize = os.path.getsize(self.filepath) if self.is_downloaded else None
        self.is_zip = self.filepath.endswith('.zip')
        # Instantiate dict
        dict.__init__(self, title=filename, fileSize=self.filesize)

    def download(self, in_parallel: bool = False, show_progress: bool = False, unzip_after: bool = False) \
            -> Union[ApplyResult, bool]:
        return self.cfolder.fs.download_file(cfile=self, local_filepath=self.filepath, in_parallel=in_parallel,
                                             show_progress=show_progress, unzip_after=unzip_after and self.is_zip)

    @property
    def folder(self) -> 'LocalFolder':
        return self.cfolder

    @folder.setter
    def folder(self, f: 'LocalFolder') -> None:
        self.cfolder = f

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.filepath) and os.path.isfile(self.filepath)

    @property
    def name(self) -> str:
        return self.filename

    @property
    def path(self) -> str:
        return self.filepath

    @property
    def size(self) -> int:
        return self.filesize


class LocalFolder(FilesystemFolder):
    """
    LocalFolder Class:
    This class, implementing `FilesystemFolder` interface, is used to transfer files from/to respective local folders.
    """

    def __init__(self, fs: 'LocalFilesystem', local_root: str, parent: Optional['LocalFolder'] = None,
                 force_create_local: bool = True):
        """
        LocalFolder class constructor.
        :param (LocalFilesystem) fs: utils.filesystems.local.LocalFilesystem instance to interact with OS kernel
        :param (str) local_root: absolute path to folder in local filesystem
        :param (optional) parent: parent folder as utils.filesystems.local.LocalFolder instance or None if root
        :param (bool) force_create_local: set to True to create local folder if it does not exist
        """
        # Save args
        self.fs = fs
        self._local_root = local_root
        # Init dict (enable json serializing of class instances)
        dict.__init__(self, local_root=self.local_root, title=os.path.basename(self.local_root) if parent else None)
        # Initially set subfolders and files to None (will be overwritten on first request of each)
        self._subfolders = []
        self._files = []
        self._parent = parent
        # Create local subfolder if requested
        if force_create_local:
            self.ensure_local_root_exists()

    @property
    def local_root(self) -> str:
        return self._local_root

    @local_root.setter
    def local_root(self, lr: str) -> None:
        self._local_root = lr

    def create_subfolder(self, folder_name: str, force_create_local: bool = True) -> 'LocalFolder':
        # Crate the folder
        return self.fs.create_folder(cloud_folder=self, folder_name=folder_name, force_create_local=force_create_local)

    def ensure_local_root_exists(self) -> None:
        if not os.path.exists(self.local_root) or not os.path.isdir(self.local_root):
            self.fs.logger.debug(f'local_root={self.local_root}: NOT FOUND/NOT DIR. Creating dir now...')
            assert 0 == os.system(f'mkdir -p "{self.local_root}"')

    def download_file(self, filename_or_cfile: Union[str, LocalFile], in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        cfile = None
        # Check type of arg
        if isinstance(filename_or_cfile, LocalFile):
            cfile = filename_or_cfile
            filename = cfile.name
        else:
            # Filename given: search inside folder for file with given filename
            filename = os.path.basename(filename_or_cfile)
            for _f in self.files:
                if filename == _f.name:
                    cfile = _f
                    break
        # If reached here and cfile is None, then no file could be found
        if not cfile:
            raise FileNotFoundError(f'"{filename}" NOT FOUND in locally-mounted Google Drive (under "{self.name}")')
        # Perform the download on the found LocalFile instance
        return cfile.download(in_parallel=in_parallel, show_progress=show_progress, unzip_after=unzip_after)

    def download(self, recursive: bool = False, in_parallel: bool = False, show_progress: bool = False,
                 unzip_after: bool = False) -> List[Union[ApplyResult, bool]]:
        _results = []
        # Download files of current folder
        for cfile in self.files:
            _results.append(
                cfile.download(in_parallel=in_parallel, show_progress=show_progress, unzip_after=unzip_after)
            )
        # If recursive=True, download the files of all subfolders
        for _sf in self.subfolders:
            _results += _sf.download(recursive=True, in_parallel=in_parallel, show_progress=show_progress,
                                     unzip_after=unzip_after)
        return _results

    @property
    def name(self) -> str:
        return os.path.basename(self.local_root)

    @property
    def files(self) -> List[LocalFile]:
        # Check if list of files has already been fetched
        if self._files:
            return self._files
        # Re-create local files list & return
        self._files = self.fs.list_files(cloud_folder=self)
        return self._files

    def file_by_name(self, filename: str) -> Optional['LocalFile']:
        for _f in self.files:
            if filename == _f.name:
                return _f
        return None

    @property
    def parent(self) -> Optional['LocalFolder']:
        return self._parent

    @parent.setter
    def parent(self, p: Optional['LocalFolder']) -> None:
        self._parent = p

    @property
    def subfolders(self) -> List['LocalFolder']:
        # Check if list of subfolders has already been fetched
        if self._subfolders:
            return self._subfolders
        # Re-create local sub-folders list & return
        self._subfolders = self.fs.list_folders(cloud_folder=self)
        return self._subfolders

    def subfolder_by_name(self, folder_name: str, recursive: bool = False) -> Optional['LocalFolder']:
        # Try find in immediate subfolders
        for _sf in self.subfolders:
            if folder_name == _sf.name:
                return _sf
        # If reached here, no subfolder found and, as so, if recursive search has been requested, re-initiate subfolders
        # search but now search the subfolders of subfolders
        if recursive:
            for _sf in self.subfolders:
                _ssf = _sf.subfolder_by_name(folder_name=folder_name, recursive=True)
                if _ssf:
                    return _ssf
        # If reached here, unfortunately no subfolder found matching the given folder name
        return None

    def subfolder_by_name_or_create(self, folder_name: str, recursive: bool = False) -> Optional['LocalFolder']:
        sf = self.subfolder_by_name(folder_name=folder_name, recursive=recursive)
        return sf if sf else \
            self.create_subfolder(folder_name=folder_name)

    def upload_file(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
                    show_progress: bool = False, is_update: bool = False) -> Union[ApplyResult, Optional[LocalFile]]:
        # Upload file using underlying LocalFilesystem instance (fake upload)
        return self.fs.upload_local_file(local_filepath=f'{self.local_root}/{os.path.basename(local_filename)}',
                                         cloud_folder=self, delete_after=delete_after, in_parallel=in_parallel,
                                         show_progress=show_progress, is_update=is_update)

    @staticmethod
    def root(capsule_or_fs: Union[LocalCapsule, 'LocalFilesystem']) -> 'LocalFolder':
        fs = capsule_or_fs if isinstance(capsule_or_fs, LocalFilesystem) else \
            LocalFilesystem(ccapsule=capsule_or_fs)
        return LocalFolder(fs=fs, local_root=fs.local_root, parent=None)


class LocalFilesystem(Filesystem):
    """
    LocalFilesystem Class:
    This class is used to interact with files stored in the locally-mounted filesystems via native OS calls.
    """

    def __init__(self, ccapsule: LocalCapsule):
        """
        LocalFilesystem class constructor.
        :param (LocalCapsule) ccapsule: a `utils.filesystems.local.LocalCapsule` instance to interact with local
                                        filesystem
        """
        # Create a thread pool for parallel uploads/downloads
        self.thread_pool = ThreadPool(processes=1)
        atexit.register(self.thread_pool.close)
        # Save args
        self.ccapsule = ccapsule
        self.local_root = ccapsule.local_root
        self.logger = ccapsule.logger

    def create_folder(self, cloud_folder: LocalFolder, folder_name: str, force_create_local: bool = False) \
            -> Optional[LocalFolder]:
        # Create a new folder in the locally-mounted Google Drive
        local_root = f'{cloud_folder.local_root}/{folder_name}'
        cloud_folder.ensure_local_root_exists()
        os.mkdir(local_root)
        # Return a new utils.filesystems.local.LocalFolder instance (for API compatibility with utils.ifaces.Filesystem)
        return LocalFolder(fs=cloud_folder.fs, local_root=local_root, parent=cloud_folder,
                           force_create_local=force_create_local)

    # noinspection DuplicatedCode
    def download_file(self, cfile: LocalFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        # If use threads, start a new thread to curry out the download and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=self.download_file_thread, args=(self, cfile),
                                                        kwds={'local_filepath': local_filepath,
                                                              'show_progress': show_progress,
                                                              'unzip_after': unzip_after})
            return async_result
        # If file exists this means the cloud file has already been downloaded and, as so, we can safely return True
        if os.path.exists(local_filepath):
            # Check if user requested unzipping of the pre-downloaded file
            if unzip_after:
                # Get list of all top-level folders in zip and check if all of them exist locally
                zip_file = zipfile.ZipFile(local_filepath, 'r')
                top_dirs = [d for d in zip_file.namelist() if d.endswith('/') and d.count('/') == 1]
                zip_file.close()
                should_unzip = False
                zip_parent_dir = os.path.dirname(local_filepath)
                for top_dir in top_dirs:
                    top_dir_abs = os.path.join(zip_parent_dir, top_dir.rstrip('/'))
                    if not os.path.exists(top_dir_abs) or not os.path.isdir(top_dir_abs):
                        should_unzip = True
                        break
                    self.logger.debug(f'[LocalCapsule::download_file] Found {top_dir_abs}')
                # Unzip
                if should_unzip:
                    assert local_filepath.endswith('.zip'), f'unzip_after=True but zip file not found: {local_filepath}'
                    unzip_result = unzip_file(zip_filepath=local_filepath)
                    if not unzip_result:
                        self.logger.critical(f'[LocalCapsule::download_file] Unzipping of {local_filepath} FAILed')
            # Regardless of the unzipping result, the download must return True, to indicate that file exists in the
            # given local filepath
            return True
        # If file not found locally, there is no such file (since this is a locally-mounted filesystem)
        return False

    @staticmethod
    def folder_cls() -> Type[LocalFolder]:
        return LocalFolder

    def list_files(self, cloud_folder: LocalFolder) -> List[LocalFile]:
        return [LocalFile(filename=_f, cfolder=cloud_folder)
                for _f in next(os.walk(cloud_folder.local_root))[2] if not _f.endswith('.cache.json')]

    def list_folders(self, cloud_folder: LocalFolder) -> List[LocalFolder]:
        return [LocalFolder(fs=self, local_root=f'{cloud_folder.local_root}/{_sf}', parent=cloud_folder)
                for _sf in next(os.walk(cloud_folder.local_root))[1]]

    def upload_local_file(self, local_filepath: str, cloud_folder: LocalFolder, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, Optional[LocalFile]]:
        # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            return self.thread_pool.apply_async(func=self.upload_local_file_thread, args=(self,),
                                                kwds={'local_filepath': local_filepath, 'cloud_folder': cloud_folder,
                                                      'delete_after': delete_after, 'show_progress': show_progress})
        # Find model name from file path
        file_basename = os.path.basename(local_filepath)
        # Create a new pydrive.files.GoogleDriveFile instance that wraps GoogleDrive API File instance
        cfile = cloud_folder.file_by_name(filename=file_basename)
        # Create return object
        uploaded_cfile = cfile if is_update and cfile else \
            LocalFile(filename=file_basename, cfolder=cloud_folder)
        return uploaded_cfile
