import atexit
import json
import os
from datetime import datetime as dt, timedelta
from multiprocessing.pool import ApplyResult, ThreadPool
from typing import Optional, List, Tuple, Union, Dict, Sequence

import httplib2
# noinspection PyProtectedMember
import torch
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
from oauth2client.client import OAuth2Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import ApiRequestError, FileNotUploadedError, FileNotDownloadableError, GoogleDriveFile
from pydrive.settings import InvalidConfigError
from requests import post as post_request

from utils.command_line_logger import CommandLineLogger
from utils.data import unzip_file
from utils.dep_free import get_tqdm
from utils.ifaces import CloudCapsule, CloudFilesystem, CloudFile, CloudFolder, CloudDataset, CloudModel


class GDriveCapsule(CloudCapsule):
    """
    GDriveCapsule Class:
    This is class is used to initiate the connection to GoogleDrive API and eventually open a TCP socket via which we
    will submit HTTP request for the various needed operations (e.g. download/upload/list ops).
    """

    def __init__(self, local_gdrive_root: str, use_http_cache: bool = False, update_credentials: bool = False):
        """
        :param (str) local_gdrive_root: absolute path to the local directory where Google Drive files will be synced to
        :param (bool) use_http_cache: set to True to have the Http client cache HTTP requests based on response headers
        :param (bool) update_credentials: set to True to have json file updated with new access_token/token expiry
                                          if a new token was generated
        """
        self.logger = CommandLineLogger(log_level='info', name=self.__class__.__name__)
        self.local_root = local_gdrive_root
        self.client_secrets_filepath = f'{local_gdrive_root}/client_secrets.json'
        self.update_credentials = update_credentials

        # Get client credentials
        self.credentials = self.get_client_dict(update_credentials=update_credentials)
        # Get authenticated Http client
        self.gservice, self.ghttp, self.gcredentials = self.get_service(
            cache_root=f'{local_gdrive_root}/.http_cache' if use_http_cache else None
        )
        # Create a pydrive.drive.GoogleAuth instance
        self.pydrive_auth = GoogleAuth()
        # and set the correct resource and http instances
        self.pydrive_auth.settings['client_config_file'] = self.client_secrets_filepath
        self.pydrive_auth.credentials = self.gcredentials
        self.pydrive_auth.http = self.ghttp
        self.pydrive_auth.service = self.gservice
        # Finally, create a pydrive.drive.GoogleDrive instance to interact with GoogleDrive API
        self.pydrive = GoogleDrive(auth=self.pydrive_auth)
        # When running locally, disable OAuthLib's HTTPs verification
        if self.client_secrets_filepath.startswith('/home/achariso'):
            os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    def get_client_dict(self, update_credentials: bool = False) -> dict:
        """
        Get updated client OAuth data, ready for instantiating google.oauth2.credentials.Credentials.
        Info on generating OAuth credentials can be found here:
        https://www.ibm.com/support/knowledgecenter/SS6KM6/com.ibm.appconnect.dev.doc/how-to-guides-for-apps/getting-oauth-credentials-for-google-applications.html
        :param update_credentials: set to True to have json file updated with new access_token/token expiry if a new
                                  token was generated
        :return: a dict object with the initial data plus the access_token, the expires_at ISO string and possibly a
                 renewed refresh_token
        """
        with open(self.client_secrets_filepath) as _fp:
            _client_dict = json.load(_fp)
            if 'web' in _client_dict.keys():
                _client_dict = _client_dict['web']

        token_expires_at = dt.fromisoformat(_client_dict['access_token_expires_at'])
        now = dt.utcnow()
        should_refresh = 'access_token' not in _client_dict.keys() or \
                         'access_token_expires_at' not in _client_dict.keys() or \
                         now >= token_expires_at or \
                         (token_expires_at - now).seconds < 300
        if not should_refresh:
            return _client_dict

        response = post_request(_client_dict['token_uri'], {
            'grant_type': 'refresh_token',
            'client_id': _client_dict['client_id'],
            'client_secret': _client_dict['client_secret'],
            'refresh_token': _client_dict['refresh_token']
        }).json()
        # print(response)
        _client_dict['access_token'] = response['access_token']
        expires_in = int(response['expires_in'])
        _client_dict['access_token_expires_at'] = (dt.utcnow() + timedelta(seconds=expires_in)).isoformat()
        _client_dict['refresh_token'] = response['refresh_token'] if 'refresh_token' in response.keys() \
            else _client_dict['refresh_token']

        # Write back to json file
        if update_credentials:
            with open(self.client_secrets_filepath, 'w') as fp:
                json.dump(_client_dict, fp, indent=4)

        return _client_dict

    def get_service(self, cache_root: Optional[str] = None) \
            -> Tuple[Optional[Resource], Optional[httplib2.Http], Optional[OAuth2Credentials]]:
        """
        Get an authenticated Google APIs resource object and http client instances ready to make requests.
        :param (optional) cache_root: absolute path to cache cache directory or None to disable cache
        :return: a tuple containing the GoogleApi service (a resource object for interacting with Google APIs), an
                 authenticated `httplib2.Http` object and an OAuth2Credentials object with the used credentials or a
                 tuple (None, None, None) in case of failure
        """
        credentials = {**self.credentials, **{
            'scopes': ['https://www.googleapis.com/auth/drive'],
            'token_expiry': dt.fromisoformat(self.credentials['access_token_expires_at']),
            'user_agent': None
        }}
        ocredentials = OAuth2Credentials(**{x: credentials[x] for x in credentials.keys() if x in [
            'access_token', 'client_id', 'client_secret', 'refresh_token',
            'token_expiry', 'token_uri', 'user_agent', 'revoke_uri',
            'id_token', 'token_response', 'scopes', 'token_info_uri', 'id_token_jwt'
        ]})

        # Create cache dir if not exists
        if cache_root is not None and not os.path.exists(cache_root):
            assert 0 == os.system(f'mkdir -p "{cache_root}"'), f'Command "mkdir -p \'{cache_root}\'": FAILed'

        # Instantiate service & Http client and return
        try:
            _http = ocredentials.authorize(httplib2.Http(cache=cache_root))
            _service = build('drive', 'v2', http=_http, cache_discovery=cache_root is not None)
            return _service, _http, ocredentials
        except KeyError or ValueError or InvalidConfigError:
            return None, None, None


class GDriveFile(CloudFile):
    """
    GDriveFile Class:
    This class, implementing `CloudFile` interface, is used to download/list info of files stored in Google Drive.
    """

    def __init__(self, pydrive_file: GoogleDriveFile, gfolder: 'GDriveFolder'):
        """
        GDriveFile class constructor.
        :param pydrive_file: an `pydrive.files.GoogleDriveFile` instance with this cloud file info
        :param gfolder: an `utils.gdrive.GDriveFolder` instance with the folder info inside of which lives this files
        """
        # Instantiate dict
        dict.__init__(self, **pydrive_file.metadata)
        # Save args
        self.pydrive_file = pydrive_file
        self.gfolder = gfolder
        # Form local file path (absolute)
        self.local_filepath = f'{self.gfolder.local_root}/{pydrive_file["title"]}'
        self.is_zip = self.local_filepath.endswith('.zip')

    def download(self, in_parallel: bool = False, show_progress: bool = False, unzip_after: bool = False) \
            -> Union[ApplyResult, bool]:
        # Check that local destination directory exists
        self.gfolder.ensure_local_root_exists()
        # Perform the actual file download
        return self.gfolder.fs.download_file(cloud_file=self, local_filepath=self.local_filepath,
                                             in_parallel=in_parallel, show_progress=show_progress,
                                             unzip_after=unzip_after and self.is_zip)

    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_filepath) and os.path.isfile(self.local_filepath)

    @property
    def name(self) -> str:
        return self.pydrive_file['title']

    @property
    def folder(self) -> 'CloudFolder':
        return self.gfolder

    @folder.setter
    def folder(self, f: 'GDriveFolder') -> None:
        self.gfolder = f


class GDriveFolder(CloudFolder):
    """
    GDriveFolder Class:
    This class, implementing `CloudFolder` interface, is used to transfer files from/to respective Google Drive folder.
    """

    ExcludedGDriveRootFolders = ['Colab Notebooks']

    def __init__(self, fs: 'GDriveFilesystem', local_root: str, cloud_root: Optional[GoogleDriveFile] = None,
                 cloud_parent: Optional['GDriveFolder'] = None, update_cache: bool = False):
        """
        GDriveFolder class constructor.
        :param (GDriveFilesystem) fs: a `utils.gdrive.GDriveFilesystem` instance to interact with the Google Drive
                                      filesystem one-level lower
        :param (str) local_root: the absolute path of the local root folder
        :param (optional) cloud_root: the data of the Google Drive root folder as a a 'pydrive.files.GoogleDriveFolder'
                                      instance or `None` for GoogleDrive's root folder
        :param (optional) cloud_parent: a 'utils.gdrive.GDriveFolder' instance that interacts with the parent folder in
                                        Google Drive given :attr:`cloud_root`
        :param (bool) update_cache: set to True to force re-fetching subfolders & files lists from GoogleDrive API and
                                    update locally-saved *.cache.json files
        """
        # Save args
        self.fs = fs
        self.cloud_root = cloud_root if cloud_root else {'id': 'root', 'title': ''}
        self.local_root = local_root
        self.cloud_parent = cloud_parent
        # Init dict (enable json serializing of class instances)
        dict.__init__(self, id=self.cloud_root['id'], local_root=self.local_root, cloud_parent=self.cloud_parent,
                      cloud_root={'id': self.cloud_root['id'], 'title': self.cloud_root['title']})
        # Initially set subfolders and files to None (will be overwritten on first request of each)
        self.cloud_subfolders = []
        self.cloud_files = []
        self.update_cache = update_cache

    def create_subfolder(self, folder_name: str) -> 'GDriveFolder':
        # Remove file cache file (if exists)
        if os.path.exists:
            os.remove(f'{self.local_root}/_subfolders.cache.json')
        self.cloud_subfolders = None
        # Crate the folder
        return self.fs.create_folder(cloud_folder=self, folder_name=folder_name)

    def ensure_local_root_exists(self) -> None:
        if not os.path.exists(self.local_root) or not os.path.isdir(self.local_root):
            self.fs.logger.warning(f'local_root={self.local_root}: NOT FOUND/NOT DIR. Creating dir now...')
            assert 0 == os.system(f'mkdir -p "{self.local_root}"')

    def download_file(self, filename_or_gfile: Union[str, GDriveFile], in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        gfile = None
        # Check type of arg
        if isinstance(filename_or_gfile, GDriveFile):
            gfile = filename_or_gfile
            filename = gfile.name
        else:
            # Filename given: search inside folder for file with given filename
            filename = filename_or_gfile
            for _f in self.files:
                if filename == _f.name:
                    gfile = _f
                    break
        # If reached here and gfile is None, then no file could be found
        if not gfile:
            raise FileNotFoundError(f'"{filename}" NOT FOUND in Google Drive (under "{self.cloud_root["title"]}")')
        # Perform the download on the found GDriveFile instance
        return gfile.download(in_parallel=in_parallel, show_progress=show_progress, unzip_after=unzip_after)

    def download(self, recursive: bool = False, in_parallel: bool = False, show_progress: bool = False,
                 unzip_after: bool = False) -> Union[ApplyResult, bool]:
        # TODO implement this method to download all the files inside this cloud folder instance
        raise NotImplementedError

    @property
    def name(self) -> str:
        assert os.path.basename(self.local_root.replace(self.fs.local_root, '')) == self.cloud_root['title']
        return self.cloud_root['title']

    @property
    def files(self) -> List[GDriveFile]:
        # Check if list of files has already been fetched
        if self.cloud_files:
            return self.cloud_files
        # Check if there exists a cached response in folder local root
        files_cache_json = f'{self.local_root}/_files.cache.json'
        if os.path.exists(files_cache_json):
            if not self.update_cache:
                with open(files_cache_json) as json_fp:
                    cloud_files_list = json.load(json_fp)
                self.cloud_files = []
                for _f in cloud_files_list:
                    _gdrive_file = GoogleDriveFile(auth=self.fs.gcapsule.pydrive_auth, metadata=_f, uploaded=True)
                    self.cloud_files.append(
                        GDriveFile(pydrive_file=_gdrive_file, gfolder=self)
                    )
                return self.cloud_files
            # Remove cache file is update_cache was set in instance
            os.remove(files_cache_json)
        # Else, fetch list, save in instance and in a file in local root and return
        cloud_files_list = self.fs.list_files(cloud_folder=self)
        self.ensure_local_root_exists()
        with open(files_cache_json, 'w') as json_fp:
            json.dump(cloud_files_list, json_fp, indent=4)
        self.cloud_files = []
        for _f in cloud_files_list:
            _gdrive_file = GoogleDriveFile(auth=self.fs.gcapsule.pydrive_auth, metadata=_f, uploaded=True)
            self.cloud_files.append(
                GDriveFile(pydrive_file=_gdrive_file, gfolder=self)
            )
        return self.cloud_files

    def file_by_name(self, filename: str) -> Optional['GDriveFile']:
        for _f in self.files:
            if filename == _f.name:
                return _f
        return None

    @property
    def parent(self) -> Optional['GDriveFolder']:
        return self.cloud_parent

    @parent.setter
    def parent(self, p: Optional['GDriveFolder']) -> None:
        self.cloud_parent = p

    @property
    def subfolders(self) -> List['GDriveFolder']:
        # Check if list of subfolders has already been fetched
        if self.cloud_subfolders:
            return self.cloud_subfolders
        # Check if there exists a cached response in folder local root
        subfolders_cache_json = f'{self.local_root}/_subfolders.cache.json'
        if os.path.exists(subfolders_cache_json):
            if not self.update_cache:
                with open(subfolders_cache_json) as json_fp:
                    cloud_subfolders_list = json.load(json_fp)
                # Process saved list in JSON and create the returned GDriveFolder list
                self.cloud_subfolders = []
                for _sf in cloud_subfolders_list:
                    _cloud_root = GoogleDriveFile(auth=self.fs.gcapsule.pydrive_auth, metadata=_sf, uploaded=True)
                    self.cloud_subfolders.append(
                        GDriveFolder(fs=self.fs, cloud_root=_cloud_root, cloud_parent=self,
                                     local_root=f'{self.local_root}/{_cloud_root["title"]}')
                    )
                return self.cloud_subfolders
            # Remove cache file is update_cache was set in instance
            os.remove(subfolders_cache_json)
        # Else, fetch list, save in instance and in a file in local root and return
        cloud_subfolders_list = self.fs.list_folders(cloud_folder=self)
        self.ensure_local_root_exists()
        with open(subfolders_cache_json, 'w') as json_fp:
            json.dump(cloud_subfolders_list, json_fp, indent=4)
        # Re-process json list
        self.cloud_subfolders = [
            GDriveFolder(fs=self.fs, local_root=f'{self.local_root}/{_sf["title"]}',
                         cloud_root=_sf, cloud_parent=self, update_cache=self.update_cache)
            for _sf in cloud_subfolders_list
        ]
        # Return freshly-fetched list of Google Drive subfolders
        return self.cloud_subfolders

    def subfolder_by_name(self, folder_name: str, recursive: bool = False) -> Optional['GDriveFolder']:
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

    def upload_file(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
                    show_progress: bool = False) -> Union[ApplyResult, Optional[GDriveFile]]:
        # Remove file cache file (if exists)
        if os.path.exists:
            os.remove(f'{self.local_root}/_files.cache.json')
        self.cloud_files = None
        # Upload file using underlying GDriveFilesystem instance
        return self.fs.upload_local_file(local_filepath=f'{self.local_root}/{os.path.basename(local_filename)}',
                                         cloud_folder=self, delete_after=delete_after, in_parallel=in_parallel,
                                         show_progress=show_progress)

    @staticmethod
    def instance(capsule_or_fs: Union[GDriveCapsule, 'GDriveFilesystem'], folder_name: str,
                 cloud_root: Optional[GoogleDriveFile] = None, cloud_parent: Optional['GDriveFolder'] = None,
                 update_cache: bool = False) -> Optional['GDriveFolder']:
        """
        Instantiate a `CloudFolder` instance to sync data between cloud and local folders.
        :param (CloudCapsule or CloudFilesystem) capsule_or_fs: a `CloudFilesystem` instance to interact with the cloud
                                                                filesystem or a 'GDriveCapsule' instance to create a
                                                                new 'CloudFilesystem' instance
        :param (str) folder_name: the name (basename) of the folder in Google Drive as well as in local filesystem
                                  (this will be appended to the :attr:`fs` local_root to form the local root of this
                                  folder instance)
        :param (GoogleDriveFile) cloud_root: a `pydrive.files.GoogleDriveFile` containing info to access the cloud
                                             folder (e.g. folder id, title, etc.)
        :param (optional) cloud_parent: an 'utils.gdrive.GDriveFolder` instance that is the parent of the folder
                                        instance to be created
        :param (bool) update_cache: set to True to delete local caches of files and subfolders and force re-fetching
                                    from cloud storage service
        :return: a `utils.gdrive.GDriveFolder` instance or None if no folder found with given name or errors occurred
        """
        fs = capsule_or_fs if isinstance(capsule_or_fs, GDriveFilesystem) else \
            GDriveFilesystem(gcapsule=capsule_or_fs)
        return GDriveFolder(fs=fs, local_root=f'{fs.local_root}/{folder_name}', cloud_root=cloud_root,
                            cloud_parent=cloud_parent, update_cache=update_cache)

    @staticmethod
    def root(capsule_or_fs: Union[GDriveCapsule, 'GDriveFilesystem'], update_cache: bool = False) -> 'GDriveFolder':
        """
        Get the an `utils.gdrive.GDriveFolder` instance to interact with Google Drive's root folder.
        :param capsule_or_fs: see `utils.ifaces.CloudFolder::instance` method
        :param update_cache: see `utils.ifaces.CloudFolder::instance` method
        :return: an `utils.gdrive.GDriveFolder` instance
        """
        fs = capsule_or_fs if isinstance(capsule_or_fs, GDriveFilesystem) else GDriveFilesystem(gcapsule=capsule_or_fs)
        return GDriveFolder(fs=fs, local_root=fs.local_root, cloud_root=None, cloud_parent=None,
                            update_cache=update_cache)


class GDriveFilesystem(CloudFilesystem):
    """
    GDriveFilesystem Class:
    This class is used to interact with files stored in Google Drive via the `google-api-client` python lib.
    """

    def __init__(self, gcapsule: GDriveCapsule):
        """
        GDriveFilesystem class constructor.
        :param (GDriveCapsule) gcapsule: a `utils.gdrive.GDriveCapsule` instance to interact with GoogleDrive filesystem
        """
        self.tqdm = get_tqdm()
        self.logger = CommandLineLogger(log_level='info', name=self.__class__.__name__)
        # Create a thread pool for parallel uploads/downloads
        self.thread_pool = ThreadPool(processes=1)
        atexit.register(self.thread_pool.close)
        # Save args
        self.gcapsule = gcapsule
        self.gservice_files = gcapsule.gservice.files()
        self.local_root = gcapsule.local_root

    def create_folder(self, cloud_folder: GDriveFolder, folder_name: str) -> Optional[GDriveFolder]:
        # Create a new pydrive.files.GoogleDriveFile instance that wraps GoogleDrive API File instance
        folder = self.gcapsule.pydrive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id': cloud_folder['id']}]
        })
        # Try creating the folder in Google Drive by submitting a "insert" request in GoogleDrive API
        try:
            folder.Upload()
            return GDriveFolder(fs=cloud_folder.fs, local_root=f'{cloud_folder.local_root}/{folder_name}',
                                cloud_root=folder, cloud_parent=cloud_folder)
        except ApiRequestError as e:
            self.logger.critical(f'Folder creation failed (folder_name={folder_name}): {str(e)}')
            return None

    def download_file(self, cloud_file: GDriveFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        # If use threads, start a new thread to curry out the download and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=self.download_file_thread,
                                                        args=(self, cloud_file, local_filepath, unzip_after))
            return async_result

        # If file exists and .download file not, this means the cloud file has already been downloaded and, as so, we
        # can safely return True
        dl_file_path = f'{local_filepath}.download'
        if os.path.exists(local_filepath) and not os.path.exists(dl_file_path):
            # Check if user requested unzipping of the pre-downloaded file
            if unzip_after:
                assert local_filepath.endswith('.zip'), f'unzip_after=True, but no zip file found at {local_filepath}'
                unzip_result = unzip_file(zip_filepath=local_filepath)
                if not unzip_result:
                    self.logger.critical(f'[GDriveCapsule::download_file] Unzipping of {local_filepath} FAILed')
            # Regardless of the unzipping result, the download must return True, to indicate that file exists in the
            # given local filepath
            return True
        # Download file from Google Drive to local root
        try:
            # Create
            file_request = self.gservice_files.get_media(fileId=cloud_file['id'])
            # Check if previous partial download exists, and read current progress
            if os.path.exists(dl_file_path) and os.path.exists(local_filepath):
                with open(dl_file_path, 'r') as dl_fp:
                    _progress = dl_fp.read().splitlines()[0]
                    if _progress and str == type(_progress) and _progress.isdigit():
                        _progress = int(_progress)
                    else:
                        _progress = 0
            else:
                _progress = 0
            # Open file pointer and begin downloading/writing bytes from Google Drive
            with open(local_filepath, 'ab' if os.path.exists(local_filepath) else 'wb') as fp:
                file_downloader = MediaIoBaseDownload(fp, file_request, chunksize=10 * 1024 * 1024)
                file_downloader._progress = _progress
                with self.tqdm(disable=not show_progress, total=100) as progress_bar:
                    done = False
                    _last_progress = 0
                    while not done:
                        mdp, done = file_downloader.next_chunk()
                        # Write next (first) byte in a .download file to be able to resume file download
                        with open(dl_file_path, 'w') as dl_fp:
                            dl_fp.write(str(mdp.resumable_progress))
                        _current_progress = int(mdp.progress() * 100)
                        progress_bar.update(_current_progress - _last_progress)
                        _last_progress = _current_progress
            os.remove(dl_file_path)
            # Check if user requested unzipping of the downloaded file
            if unzip_after:
                assert local_filepath.endswith('.zip'), f'unzip_after=True, but no zip file found at {local_filepath}'
                unzip_result = unzip_file(zip_filepath=local_filepath)
                if not unzip_result:
                    self.logger.critical(f'[GDriveCapsule::download_file] Unzipping of {local_filepath} FAILed')
            # Regardless of the unzipping result, the download has been completed successfully and, as so, we must
            # return True
            return True
        except ApiRequestError or FileNotUploadedError or FileNotDownloadableError as e:
            self.logger.critical(f'[GDriveFilesystem::download_file] {str(e)}')
            return False

    def list_files(self, cloud_folder: GDriveFolder) -> List[GoogleDriveFile]:
        gdrive_query = f"'{cloud_folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' " + \
                       f"and trashed=false"
        return self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()

    def list_folders(self, cloud_folder: GDriveFolder) -> List[GoogleDriveFile]:
        gdrive_query = f"'{cloud_folder['id']}' in parents and mimeType = 'application/vnd.google-apps.folder' " + \
                       f"and trashed=false"
        return self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()

    def upload_local_file(self, local_filepath: str, cloud_folder: GDriveFolder, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, Optional[GDriveFile]]:
        # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=GDriveFilesystem.upload_local_file_thread,
                                                        args=(self, local_filepath, cloud_folder, delete_after))
            return async_result

        # Find model name from file path
        file_basename = os.path.basename(local_filepath)
        # Create a new pydrive.files.GoogleDriveFile instance that wraps GoogleDrive API File instance
        file = self.gcapsule.pydrive.CreateFile({
            'title': file_basename,
            'parents': [{'id': cloud_folder['id']}]
        })
        # Load local file data into the File instance
        assert os.path.exists(local_filepath)
        file.SetContentFile(local_filepath)
        try:
            file.Upload()
        except ApiRequestError as e:
            self.logger.critical(f'File upload failed (local_filepath={local_filepath}): {str(e)}')
            return None
        # Close file handle
        file.content.close()
        # Delete file after successful upload
        if delete_after:
            try:
                os.remove(local_filepath)
            except FileNotFoundError or AttributeError as e:
                self.logger.critical(f'Could not remove local file (at {local_filepath}): {str(e)}')
                return None
        return GDriveFile(pydrive_file=file, gfolder=cloud_folder)


class GDriveDataset(CloudDataset):
    """
    GDriveDataset Class:
    This class is used to transfer dataset from Google Drive to local file system and unzip it to be able to use it in
    a data loader afterwards.
    """

    def __init__(self, dataset_gfolder: GDriveFolder, zip_filename: str):
        """
        GDriveDataset class constructor.
        :param (GDriveFolder) dataset_gfolder: a `utils.gdrive.GDriveFolder` instance to interact with dataset folder
                                               in Google Drive
        :param (str) zip_filename: the name of dataset's main .zip file (should be inside Google Drive folder root)
        """
        self.dataset_gfolder = dataset_gfolder
        self.zip_filename = zip_filename
        self.zip_gfile = self.dataset_gfolder.file_by_name(zip_filename)
        assert self.zip_gfile is not None, f'zip_filename={zip_filename} NOT FOUND in Google Drive folder root'

    def fetch_and_unzip(self, in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        return self.dataset_gfolder.download_file(filename_or_gfile=self.zip_gfile, in_parallel=in_parallel,
                                                  show_progress=show_progress, unzip_after=True)

    def is_fetched_and_unzipped(self) -> bool:
        zip_local_filepath = f'{self.dataset_gfolder.local_root}/{self.zip_filename}'
        dataset_local_path = zip_local_filepath.replace('.zip', '')
        return os.path.exists(zip_local_filepath) and os.path.exists(dataset_local_path) and \
               os.path.isfile(zip_local_filepath) and os.path.isdir(dataset_local_path)

    @staticmethod
    def instance(groot_or_capsule_or_fs: Union[GDriveFolder, GDriveCapsule, GDriveFilesystem],
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
        if isinstance(groot_or_capsule_or_fs, GDriveFolder):
            groot = groot_or_capsule_or_fs
        else:
            fs = groot_or_capsule_or_fs if isinstance(groot_or_capsule_or_fs, GDriveFilesystem) else \
                GDriveFilesystem(gcapsule=groot_or_capsule_or_fs)
            groot = GDriveFolder.root(capsule_or_fs=fs)
        # Find the Google Drive folder instance that corresponds to the dataset with the given folder name
        dataset_gfolder = groot.subfolder_by_name(folder_name=dataset_folder_name, recursive=True)
        if not dataset_gfolder:
            return None
        # Instantiate a GDriveDataset object with the found Google Drive folder instance
        return GDriveDataset(dataset_gfolder=dataset_gfolder, zip_filename=zip_filename)


class GDriveModel(CloudModel):
    """
    GDriveModel Class:
    In the Google Drive the directory structure should be as follows:

    [root]  Models
            ├── model_name=<model_name: str>: checkpoints for model named after `model_name`
            │   ├── Checkpoints
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
            │  ...
            │   │
            │   └── Metrics
            │       ├── <batch_size>_<step_min>_<step_max>.jpg
            │       ├── <batch_size>_<another_step_min>_<another_step_max>.jpg
            │       ├──                    ...
            │       ├── <another_batch_size>_<step_min>_<step_max>.jpg
            │       └── <another_batch_size>_<another_step_min>_<another_step_max>.jpg
           ...
            │
            └── model_name=<another_model_name>: checkpoints for model named after `another_model_name`
                    ├── Checkpoints
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
                    └── Metrics
                        ├── <batch_size>_<step_min>_<step_max>.jpg
                        ├── <batch_size>_<another_step_min>_<another_step_max>.jpg
                        ├──                    ...
                        ├── <another_batch_size>_<step_min>_<step_max>.jpg
                        └── <another_batch_size>_<another_step_min>_<another_step_max>.jpg

    Based on this directory structure, this class is used to download/upload model checkpoints to Google Drive and check
    if a checkpoint at a given batch size/step combination is present in the local filesystem.
    """

    def __init__(self, model_gfolder: GDriveFolder):
        """
        GDriveModel class constructor.
        :param (GDriveFolder) model_gfolder: a `utils.gdrive.GDriveFolder` instance to interact with model folder in
                                             Google Drive filesystem
        """
        self.logger = CommandLineLogger(log_level='info')
        # Save args
        self.gfolder = model_gfolder
        self.local_chkpts_root = model_gfolder.local_root
        # Define extra properties
        self.chkpts_gfolder = model_gfolder.subfolder_by_name(folder_name='Checkpoints')
        self.metrics_gfolder = model_gfolder.subfolder_by_name(folder_name='Metrics')
        self.model_name = model_gfolder.local_root.split(sep='=', maxsplit=1)[-1]
        # Get all checkpoint folders (i.e. folders with names "batch_size=<batch_size>")
        self.chkpts_batch_gfolders = {}
        for _sf in self.chkpts_gfolder.subfolders:
            if _sf.name.startswith('batch_size='):
                self.chkpts_batch_gfolders[int(_sf.name.replace('batch_size=', ''))] = _sf
        # No batch_size=* subfolders found: create a batch checkpoint folder with hypothetical batch_size=-1
        self.chkpts_batch_gfolders[-1] = self.chkpts_gfolder

    def ensure_batch_gfolder_exists(self, batch_size: Optional[int] = None) -> None:
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
                self.chkpts_gfolder.create_subfolder(folder_name=f'batch_size={str(batch_size)}')

    def download_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None,
                            in_parallel: bool = False, show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Check & correct given args
        if batch_size is None:
            batch_size = -1
        step_str = step if isinstance(step, str) else str(step).zfill(10)
        # Ensure folder exists locally & in Google Drive
        self.ensure_batch_gfolder_exists(batch_size=batch_size)
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

    def fetch_checkpoint(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
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
            return False
        chkpts_gfolder = self.chkpts_batch_gfolders[batch_size]
        # Format and return
        return f'{chkpts_gfolder.local_root}/{step_str}.pth'

    def is_checkpoint_fetched(self, step: Union[int, str], batch_size: Optional[int] = None) -> str or False:
        local_filepath = self._get_checkpoint_filepath(step=step, batch_size=batch_size)
        return local_filepath if os.path.exists(local_filepath) and os.path.isfile(local_filepath) \
            else False

    def list_checkpoints(self, batch_size: Optional[int] = None, only_keys: Optional[Sequence[str]] = None) \
            -> List[GDriveFile or dict]:
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

    def list_all_checkpoints(self, only_keys: Optional[Sequence[str]] = None) -> Dict[int, List[GDriveFile or dict]]:
        _return_dict = {}
        for batch_size in self.chkpts_batch_gfolders.keys():
            _return_dict[batch_size] = self.list_checkpoints(batch_size=batch_size, only_keys=only_keys)
        return _return_dict

    def save_and_upload_checkpoint(self, state_dict: dict, step: Union[int, str], batch_size: Optional[int] = None,
                                   delete_after: bool = False, in_parallel: bool = False,
                                   show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Get new checkpoint file name
        new_chkpt_path = self._get_checkpoint_filepath(step=step, batch_size=batch_size)
        # Save state_dict locally
        torch.save(state_dict, new_chkpt_path)
        # Upload new checkpoint file to Google Drive
        return self.upload_checkpoint(chkpt_filename=new_chkpt_path, delete_after=delete_after,
                                      in_parallel=in_parallel, show_progress=show_progress)

    def upload_checkpoint(self, chkpt_filename: str, delete_after: bool = False, in_parallel: bool = False,
                          show_progress: bool = False) -> Union[ApplyResult, bool]:
        # Extract data from filename
        chkpt_filename = os.path.basename(chkpt_filename)
        chkpt_filename_parts = chkpt_filename.split(sep='_')
        assert len(chkpt_filename_parts) in [2., 3]
        assert chkpt_filename_parts[0] == self.model_name
        step = chkpt_filename_parts[1]
        step = int(step) if step.isdigit() else step
        batch_size = chkpt_filename_parts[2] if len(chkpt_filename_parts) == 3 else None
        # Check if needed to create the folder in Google Drive before uploading
        if batch_size is None:
            batch_size = -1
        # Ensure folder exists locally & in Google Drive
        self.ensure_batch_gfolder_exists(batch_size=batch_size)
        # Upload local file to Google Drive
        upload_gfolder = self.chkpts_batch_gfolders[batch_size]
        return upload_gfolder.upload_file(local_filename=chkpt_filename, delete_after=delete_after,
                                          in_parallel=in_parallel, show_progress=show_progress)


if __name__ == '__main__':
    import time

    _start_time = time.time()
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _gcapsule = GDriveCapsule(local_gdrive_root=_local_gdrive_root, use_http_cache=True, update_credentials=True)
    _fs = GDriveFilesystem(gcapsule=_gcapsule)
    _groot = GDriveFolder.root(capsule_or_fs=_fs)
    # print(json.dumps(_groot.subfolders, indent=4))
    # print('')
    # print('----------------------')
    # print('')
    #
    # print(json.dumps(_groot.subfolder_by_name('Model Checkpoints'), indent=4))
    # print('')
    # print('----------------------')
    # print('')
    #
    # df_icrb_gfolder = _groot.subfolder_by_name('In-shop Clothes Retrieval Benchmark', recursive=True)
    # print(json.dumps(df_icrb_gfolder, indent=4) if df_icrb_gfolder else df_icrb_gfolder)
    # print('')
    # print('----------------------')
    # print('')
    #
    # look_book_gfolder = _groot.subfolder_by_name('LookBook', recursive=True)
    # print(json.dumps(look_book_gfolder, indent=4) if look_book_gfolder else look_book_gfolder)
    # print('')
    # print('----------------------')
    # print('')
    # print(json.dumps(look_book_gfolder.files, indent=4) if look_book_gfolder else look_book_gfolder)
    # print('')
    # print('----------------------')
    # print('')

    # from datasets.look_book import PixelDTDataset
    #
    # _start_time = time.time()
    # pixel_dt_dataset = PixelDTDataset(dataset_gfolder_or_groot=_groot)
    # print("--- %s seconds ---" % (time.time() - _start_time))
    # print(pixel_dt_dataset.is_fetched_and_unzipped())

    from modules.inception import InceptionV3

    inception = InceptionV3(model_gfolder_or_groot=_groot)
    # print(json.dumps(inception.list_all_checkpoints(), indent=4))
    print(inception.fetch_latest_checkpoint())
    print("--- %s seconds ---" % (time.time() - _start_time))

    model_name = 'pgpg'
    gmodel = GDriveModel(model_gfolder=_groot.subfolder_by_name(folder_name=f'model_name={model_name}', recursive=True))
    print(gmodel.model_name)
    # NOT RUN: print(gmodel.fetch_latest_checkpoint(batch_size=48))
    # _start_time = time.time()
    # print(json.dumps(gmodel.list_checkpoints(batch_size=48), indent=4))
    # print("--- %s seconds ---" % (time.time() - _start_time))
    # print(json.dumps(gmodel.list_checkpoints(batch_size=48, only_keys=('id', 'title', 'mimeType')), indent=4))
    print("--- %s seconds ---" % (time.time() - _start_time))
