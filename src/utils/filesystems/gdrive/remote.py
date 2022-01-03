import atexit
import json
import os
import sys
import time
from datetime import datetime as dt, timedelta
from io import BufferedWriter, BufferedRandom
from multiprocessing.pool import ApplyResult, ThreadPool
from typing import Optional, List, Tuple, Union, Type, TextIO

import httplib2
from apiclient import errors as pydrive_errors
# noinspection PyProtectedMember
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
# noinspection PyProtectedMember
from googleapiclient.http import MediaIoBaseDownload, _retry_request, HttpRequest, MediaIoBaseUpload
from oauth2client.client import OAuth2Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import ApiRequestError, FileNotUploadedError, FileNotDownloadableError, GoogleDriveFile
from pydrive.settings import InvalidConfigError
from requests import post as post_request

from utils.command_line_logger import CommandLineLogger
from utils.data import unzip_file
from utils.dep_free import get_tqdm
from utils.ifaces import FilesystemCapsule, Filesystem, FilesystemFile, FilesystemFolder


class GDriveCapsule(FilesystemCapsule):
    """
    GDriveCapsule Class:
    This is class is used to initiate the connection to GoogleDrive API and eventually open a TCP socket via which we
    will submit HTTP request for the various needed operations (e.g. download/upload/list ops).
    """

    def __init__(self, local_gdrive_root: str, use_refresh_token: bool = False, use_http_cache: bool = False,
                 update_credentials: bool = False, project_root: str = '/'):
        """
        :param (str) local_gdrive_root: absolute path to the local directory where Google Drive files will be synced to
        :param (bool) use_refresh_token: set to True to use refresh token found inside the `client_secrets.json` file,
                                         else a prompt with an authorization url will be displayed.
                                         Attention: using `refresh_token` multiple times is known to cause 403 errors!
        :param (bool) use_http_cache: set to True to have the Http client cache HTTP requests based on response headers
        :param (bool) update_credentials: set to True to have json file updated with new access_token/token expiry
                                          if a new token was generated
        :param (str) project_root: relative path to folder considered as root for the project
        """
        self.logger = CommandLineLogger(log_level=os.getenv('TRAIN_LOG_LEVEL', 'info'), name=self.__class__.__name__)
        self.local_root = f'{local_gdrive_root}{project_root}'.rstrip('/')
        self.client_secrets_filepath = f'{self.local_root}/client_secrets.json'
        self.update_credentials = update_credentials

        # Create a pydrive.drive.GoogleAuth instance
        self.pydrive_auth = GoogleAuth()
        self.pydrive_auth.settings['client_config_file'] = self.client_secrets_filepath

        if use_refresh_token:
            # Get client credentials
            self.credentials = self.get_client_dict(update_credentials=update_credentials)
            # Get authenticated Http client
            self.gservice, self.ghttp, self.gcredentials = self.get_service(
                cache_root=f'{self.local_root}/.http_cache' if use_http_cache else None
            )
            # and set the correct resource and http instances
            self.pydrive_auth.credentials = self.gcredentials
            self.pydrive_auth.http = self.ghttp
            self.pydrive_auth.service = self.gservice
        else:
            # Authenticate using authorization url (Colab-like authentication flow)
            auth_url = self.pydrive_auth.GetAuthUrl()
            self.logger.critical(f'Get the authorization code from: {auth_url}')
            time.sleep(0.5)
            self.pydrive_auth.Auth(input('code = ').strip())
            # Save service and http objects in instance
            self.gservice = self.pydrive_auth.service
            self.ghttp = self.pydrive_auth.http

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

        if 'access_token_expires_at' in _client_dict.keys():
            token_expires_at = dt.fromisoformat(_client_dict['access_token_expires_at'])
        else:
            token_expires_at = dt.fromtimestamp(0.0)
        now = dt.utcnow()
        should_refresh = 'access_token' not in _client_dict.keys() or \
                         'access_token_expires_at' not in _client_dict.keys() or \
                         now >= token_expires_at or \
                         (token_expires_at - now).seconds < 300
        self.logger.debug(json.dumps({
            'now': str(now),
            'access_token_expires_at': str(token_expires_at),
            'cond1': now >= token_expires_at,
            'cond2': ((token_expires_at - now).seconds, (token_expires_at - now).seconds < 300),
            'should_refresh': should_refresh,
        }, indent=4))
        if not should_refresh:
            return _client_dict

        response = post_request(_client_dict['token_uri'], {
            'grant_type': 'refresh_token',
            'client_id': _client_dict['client_id'],
            'client_secret': _client_dict['client_secret'],
            'refresh_token': _client_dict['refresh_token']
        }).json()
        self.logger.debug('[get_client_dict] Response: ' + json.dumps(response, indent=4))
        _client_dict['access_token'] = response['access_token']
        expires_in = int(response['expires_in'])
        _client_dict['access_token_expires_at'] = (dt.utcnow() + timedelta(seconds=expires_in)).isoformat()
        _client_dict['refresh_token'] = response['refresh_token'] if 'refresh_token' in response.keys() \
            else _client_dict['refresh_token']

        # Write back to json file
        if update_credentials:
            with open(self.client_secrets_filepath, 'w') as fp:
                json.dump({'web': _client_dict} if 'web' not in _client_dict.keys() else _client_dict, fp, indent=4)

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


class GDriveFile(FilesystemFile):
    """
    GDriveFile Class:
    This class, implementing `FilesystemFile` interface, is used to download/list info of files stored in Google Drive.
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
        return self.gfolder.fs.download_file(gfile=self, local_filepath=self.local_filepath,
                                             in_parallel=in_parallel, show_progress=show_progress,
                                             unzip_after=unzip_after and self.is_zip)

    @property
    def folder(self) -> 'FilesystemFolder':
        return self.gfolder

    @folder.setter
    def folder(self, f: 'GDriveFolder') -> None:
        self.gfolder = f

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.local_filepath) and os.path.isfile(self.local_filepath)

    @property
    def name(self) -> str:
        return self.pydrive_file['title']

    @property
    def path(self) -> str:
        return self.local_filepath

    @property
    def size(self) -> int:
        return int(self.pydrive_file['fileSize'])


class GDriveFolder(FilesystemFolder):
    """
    GDriveFolder Class:
    This class, implementing `utils.ifaces.FilesystemFolder` interface, is used to transfer files from/to the respective
    Google Drive folder.
    """

    ExcludedGDriveRootFolders = ['Colab Notebooks']

    def __init__(self, fs: 'GDriveFilesystem', local_root: str, cloud_root: Optional[GoogleDriveFile] = None,
                 cloud_parent: Optional['GDriveFolder'] = None, update_cache: bool = False,
                 force_create_local: bool = False):
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
        :param (bool) force_create_local: set to True to create local folder immediately after creating folder in cloud
                                          storage; the local folder is created if not exists before any file upload and
                                          download
        """
        # Save args
        self.fs = fs
        self.cloud_root = cloud_root if cloud_root else {'id': 'root', 'title': ''}
        self._local_root = local_root
        self.cloud_parent = cloud_parent
        # Init dict (enable json serializing of class instances)
        dict.__init__(self, id=self.cloud_root['id'], local_root=self.local_root, cloud_parent=self.cloud_parent,
                      cloud_root={'id': self.cloud_root['id'], 'title': self.cloud_root['title']})
        # Initially set subfolders and files to None (will be overwritten on first request of each)
        self.cloud_subfolders = []
        self.cloud_files = []
        self.update_cache = update_cache
        # Create local subfolder if requested
        if force_create_local:
            self.ensure_local_root_exists()

    @property
    def local_root(self) -> str:
        return self._local_root

    @local_root.setter
    def local_root(self, lr: str) -> None:
        self._local_root = lr

    def refresh_files_cache(self) -> None:
        """
        Fetches the file list using `pydrive`, saves the response to a _files.cache.json file locally and assigns newly
        created `utils.gdrive.GDriveFile` instances to `self.cloud_files`.
        """
        files_cache_json = f'{self.local_root}/_files.cache.json'
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

    def clear_files_cache(self, clear_instance: bool = True) -> None:
        """
        Clears the `_files.cache.json` inside the folder local root.
        :param (bool) clear_instance: set to True to erase the GDriveFile objects in `self`
        """
        files_cache_json = f'{self.local_root}/_files.cache.json'
        if os.path.exists(files_cache_json):
            os.remove(files_cache_json)
        if clear_instance:
            self.cloud_files = None

    def create_subfolder(self, folder_name: str, force_create_local: bool = False) -> 'GDriveFolder':
        # Remove file cache file (if exists)
        if os.path.exists(f'{self.local_root}/_subfolders.cache.json'):
            os.remove(f'{self.local_root}/_subfolders.cache.json')
        self.cloud_subfolders = None
        # Crate the folder
        return self.fs.create_folder(cloud_folder=self, folder_name=folder_name, force_create_local=force_create_local)

    def ensure_local_root_exists(self) -> None:
        if not os.path.exists(self.local_root) or not os.path.isdir(self.local_root):
            self.fs.logger.debug(f'local_root={self.local_root}: NOT FOUND/NOT DIR. Creating dir now...')
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
                 unzip_after: bool = False) -> List[Union[ApplyResult, bool]]:
        _results = []
        # Download files of current folder
        for gfile in self.files:
            _results.append(
                gfile.download(in_parallel=in_parallel, show_progress=show_progress, unzip_after=unzip_after)
            )
        # If recursive=True, download the files of all subfolders
        for _sf in self.subfolders:
            _results += _sf.download(recursive=True, in_parallel=in_parallel, show_progress=show_progress,
                                     unzip_after=unzip_after)
        return _results

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
            self.clear_files_cache()
        # Else, fetch list, save in instance and in a file in local root and return
        self.refresh_files_cache()
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

    def subfolder_by_name_or_create(self, folder_name: str, recursive: bool = False) -> Optional['GDriveFolder']:
        sf = self.subfolder_by_name(folder_name=folder_name, recursive=recursive)
        return sf if sf else \
            self.create_subfolder(folder_name=folder_name)

    def upload_file(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
                    show_progress: bool = False, is_update: bool = False) -> Union[ApplyResult, Optional[GDriveFile]]:
        # Remove file cache file (if exists)
        if not is_update:
            self.clear_files_cache()
        # Upload file using underlying GDriveFilesystem instance
        return self.fs.upload_local_file(local_filepath=f'{self.local_root}/{os.path.basename(local_filename)}',
                                         cloud_folder=self, delete_after=delete_after, in_parallel=in_parallel,
                                         show_progress=show_progress, is_update=is_update)

    @staticmethod
    def instance(capsule_or_fs: Union[GDriveCapsule, 'GDriveFilesystem'], folder_name: str,
                 cloud_root: Optional[GoogleDriveFile] = None, cloud_parent: Optional['GDriveFolder'] = None,
                 update_cache: bool = False) -> Optional['GDriveFolder']:
        """
        Instantiate a `FilesystemFolder` instance to sync data between cloud and local folders.
        :param (FilesystemCapsule or Filesystem) capsule_or_fs: a `Filesystem` instance to interact with the cloud
                                                                filesystem or a 'GDriveCapsule' instance to create a
                                                                new 'Filesystem' instance
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
    def root(capsule_or_fs: Union[GDriveCapsule, 'GDriveFilesystem'], update_cache: bool = False,
             cloud_root: Optional[Union[GoogleDriveFile, str]] = None) -> 'GDriveFolder':
        """
        Get the an `utils.gdrive.GDriveFolder` instance to interact with Google Drive's root folder.
        :param capsule_or_fs: see `utils.ifaces.FilesystemFolder::instance` method
        :param update_cache: see `utils.ifaces.FilesystemFolder::instance` method
        :param cloud_root: either a string with the relative path to the new root or a pydrive.files.GoogleDriveFile
                           instance with the cloud folder to be considered as Google Drive root
        :return: an `utils.gdrive.GDriveFolder` instance
        """
        fs = capsule_or_fs if isinstance(capsule_or_fs, GDriveFilesystem) else GDriveFilesystem(gcapsule=capsule_or_fs)
        cloud_root_ob = cloud_root if type(cloud_root) == GoogleDriveFile else \
            fs.gcapsule.pydrive.CreateFile({'id': cloud_root})
        gf = GDriveFolder(fs=fs, local_root=fs.local_root, cloud_root=cloud_root_ob, update_cache=update_cache)
        if cloud_root is not None:
            gf.cloud_root['title'] = 'None'
        ### Set correct root via relative path
        # if type(cloud_root) == str:
        #     folder_names = cloud_root.strip('/').split('/')
        #     sf, spf = gf, None
        #     for fn in folder_names:
        #         sf, spf = sf.subfolder_by_name(folder_name=fn, recursive=False), sf
        #         assert sf is not None
        #     return sf
        return gf


class GMediaIoDownload(MediaIoBaseDownload):
    """
    GMediaIoDownload Class:
    This class inherits from `googleapiclient.http.MediaIoBaseDownload` to just re-define the return type of its method
    `next_chunk()` and enable better display of download progress using the `tqdm` lib.
    """

    # By default, the files will be downloaded in 10MB chunks
    DefaultChunkSizeInMB = 1

    def __init__(self, fp: Union[BufferedWriter, BufferedRandom, TextIO], file_request: HttpRequest,
                 bytes_written: int = 0):
        """
        GMediaIoDownload class constructor.
        :param (TextIO) fp: the binary file pointer (e.g. the result of calling `open('<file_path>', 'wb')`
        :param (HttpRequest) file_request: the media request as a `googleapiclient.http.HttpRequest` object
        :param (int) bytes_written: current file length (the file name should end in .download); this is used for
                     resumable file downloads
        """
        super(GMediaIoDownload, self).__init__(fd=fp, request=file_request,
                                               chunksize=self.DefaultChunkSizeInMB * 1024 * 1024)
        if bytes_written > 0:
            self._progress = bytes_written

    def next_chunk(self, num_retries=0):
        """Get the next chunk of the download.
        Args:
            num_retries: Integer, number of times to retry with randomized
                exponential backoff. If all retries fail, the raised HttpError
                represents the last request. If zero (default), we attempt the
                request only once.

        Returns:
            (status, done): (MediaDownloadProgress, boolean)
             The value of 'done' will be True when the media has been fully
             downloaded or the total size of the media is unknown.

        Raises:
            googleapiclient.errors.HttpError if the response was not a 2xx.
            httplib2.HttpLib2Error if a transport error has occurred.
        """
        headers = self._headers.copy()
        headers["range"] = "bytes=%d-%d" % (
            self._progress,
            self._progress + self._chunksize,
        )
        http = self._request.http
        resp, content = _retry_request(http, num_retries, "media download", self._sleep, self._rand, self._uri, "GET",
                                       headers=headers)
        if resp.status in [200, 206]:
            if "content-location" in resp and resp["content-location"] != self._uri:
                self._uri = resp["content-location"]
            self._progress += len(content)
            bytes_count = self._fd.write(content)

            if "content-range" in resp:
                content_range = resp["content-range"]
                length = content_range.rsplit("/", 1)[1]
                self._total_size = int(length)
            elif "content-length" in resp:
                self._total_size = int(resp["content-length"])

            if self._total_size is None or self._progress == self._total_size:
                self._done = True
            return {'bytes_written': bytes_count, 'total': self._total_size}, self._done
        elif resp.status == 416:
            # 416 is Range Not Satisfiable
            # This typically occurs with a zero byte file
            content_range = resp["content-range"]
            length = content_range.rsplit("/", 1)[1]
            self._total_size = int(length)
            if self._total_size == 0:
                self._done = True
                return {'bytes_written': 0, 'total': self._total_size}, self._done
        raise HttpError(resp, content, uri=self._uri)


class GDriveFilesystem(Filesystem):
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
        self.logger = gcapsule.logger
        # Create a thread pool for parallel uploads/downloads
        self.thread_pool = ThreadPool(processes=1)
        atexit.register(self.thread_pool.close)
        # Save args
        self.gcapsule = gcapsule
        self.gservice_files = gcapsule.gservice.files()
        self.local_root = gcapsule.local_root

    def create_folder(self, cloud_folder: GDriveFolder, folder_name: str, force_create_local: bool = False) \
            -> Optional[GDriveFolder]:
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
                                cloud_root=folder, cloud_parent=cloud_folder, force_create_local=force_create_local)
        except ApiRequestError as e:
            self.logger.critical(f'Folder creation failed (folder_name={folder_name}): {str(e)}')
            return None

    # noinspection DuplicatedCode
    def download_file(self, gfile: GDriveFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False, unzip_after: bool = False) -> Union[ApplyResult, bool]:
        # If use threads, start a new thread to curry out the download and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=self.download_file_thread, args=(self, gfile),
                                                        kwds={'local_filepath': local_filepath,
                                                              'show_progress': show_progress,
                                                              'unzip_after': unzip_after})
            return async_result

        # If file exists this means the cloud file has already been downloaded and, as so, we can safely return True
        if os.path.exists(local_filepath):
            # Check if user requested unzipping of the pre-downloaded file
            if unzip_after and not os.path.exists(local_filepath.replace('.zip', '')):
                assert local_filepath.endswith('.zip'), f'unzip_after=True, but no zip file found at {local_filepath}'
                unzip_result = unzip_file(zip_filepath=local_filepath)
                if not unzip_result:
                    self.logger.critical(f'[GDriveCapsule::download_file] Unzipping of {local_filepath} FAILed')
            # Regardless of the unzipping result, the download must return True, to indicate that file exists in the
            # given local filepath
            return True
        # Download file from Google Drive to local root
        try:
            # Create HTTP request to GoogleDrive API
            file_request = self.gservice_files.get_media(fileId=gfile['id'])
            # Check if previous partial download exists, and read current progress
            dl_filepath = f'{local_filepath}.download'
            if os.path.exists(dl_filepath):
                bytes_written = os.path.getsize(dl_filepath)
            else:
                bytes_written = 0
            # Open file pointer and begin downloading/writing bytes from Google Drive
            with open(dl_filepath, 'ab' if os.path.exists(dl_filepath) else 'wb') as fp:
                file_downloader = GMediaIoDownload(fp=fp, file_request=file_request, bytes_written=bytes_written)
                # If show_progress=True, will initialize the tqdm for file download using the code provided here:
                # https://github.com/sirbowen78/lab/blob/master/file_handling/dl_file1.py
                # with self.tqdm(disable=not show_progress, total=100, unit='%') as progress_bar:
                with self.tqdm(
                        disable=not show_progress,
                        unit="B",  # unit string to be displayed.
                        unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
                        unit_divisor=1024,  # is used when unit_scale is true
                        total=gfile.size,  # the total iteration.
                        initial=bytes_written,
                        file=sys.stdout,
                        desc=gfile.name  # prefix to be displayed on progress bar.
                ) as progress_bar:
                    done = False
                    while not done:
                        # Get next file chunk
                        result_dict, done = file_downloader.next_chunk()
                        # Update progress bar
                        progress_bar.update(result_dict['bytes_written'])
            # After file download finishes, rename .download file to given filename
            os.rename(dl_filepath, local_filepath)
            # Check if user requested unzipping of the downloaded file
            if unzip_after and not os.path.exists(local_filepath.replace('.zip', '')):
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

    @staticmethod
    def folder_cls() -> Type[GDriveFolder]:
        return GDriveFolder

    def list_files(self, cloud_folder: GDriveFolder) -> List[GoogleDriveFile]:
        gdrive_query = f"'{cloud_folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' " + \
                       f"and trashed=false"
        return self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()

    def list_folders(self, cloud_folder: GDriveFolder) -> List[GoogleDriveFile]:
        gdrive_query = f"'{cloud_folder['id']}' in parents and mimeType = 'application/vnd.google-apps.folder' " + \
                       f"and trashed=false"
        return self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()

    def upload_local_file(self, local_filepath: str, cloud_folder: GDriveFolder, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False, is_update: bool = False) \
            -> Union[ApplyResult, Optional[GDriveFile]]:
        # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            return self.thread_pool.apply_async(func=self.upload_local_file_thread, args=(self,),
                                                kwds={'local_filepath': local_filepath, 'cloud_folder': cloud_folder,
                                                      'delete_after': delete_after, 'show_progress': show_progress})
        # Find model name from file path
        file_basename = os.path.basename(local_filepath)
        # Create a new pydrive.files.GoogleDriveFile instance that wraps GoogleDrive API File instance
        gfile = cloud_folder.file_by_name(filename=file_basename)
        if is_update and gfile:
            cloud_folder.clear_files_cache(clear_instance=False)
            file = gfile.pydrive_file
        else:
            file = self.gcapsule.pydrive.CreateFile({
                'title': file_basename,
                'parents': [{'id': cloud_folder['id']}]
            })
        # Load local file data into the File instance
        assert os.path.exists(local_filepath), f'local_filepath={local_filepath}'
        file.SetContentFile(local_filepath)
        # Set mime type
        if file.get('mimeType') is None:
            file['mimeType'] = 'application/octet-stream'

        # Upload file
        try:
            # Create upload HTTP request object
            if is_update:
                http_request = self.gservice_files.update(
                    body=file.GetChanges(),
                    media_body=MediaIoBaseUpload(file.content, file['mimeType'],
                                                 chunksize=GMediaIoDownload.DefaultChunkSizeInMB * 1024 * 1024,
                                                 resumable=True),
                    fileId=file['id']
                )
            else:
                http_request = self.gservice_files.insert(
                    body=file.GetChanges(),
                    media_body=MediaIoBaseUpload(file.content, file['mimeType'],
                                                 chunksize=GMediaIoDownload.DefaultChunkSizeInMB * 1024 * 1024,
                                                 resumable=True)
                )

            # Check if should resume upload
            if os.path.exists(f'{local_filepath}.part'):
                with open(f'{local_filepath}.part', 'r') as part_fp:
                    http_request_data = json.load(part_fp)
                http_request.resumable_uri = http_request_data['resumable_uri']

            # Start resumable file upload
            # If show_progress=True, will initialize the tqdm for file download using the code provided here:
            # https://github.com/sirbowen78/lab/blob/master/file_handling/dl_file1.py
            # with self.tqdm(disable=not show_progress, total=100, unit='%') as progress_bar:
            with self.tqdm(
                    disable=not show_progress,
                    unit="B",  # unit string to be displayed.
                    unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
                    unit_divisor=1024,  # is used when unit_scale is true
                    total=http_request.resumable.size(),
                    initial=http_request.resumable_progress,
                    file=sys.stdout,
                    desc=f'Uploading "{file_basename}"'  # prefix to be displayed on progress bar.
            ) as progress_bar:

                metadata = None
                progress_last = 0
                while metadata is None:
                    # Upload next chunk to Google Drive
                    progress, metadata = http_request.next_chunk(http=self.gcapsule.ghttp, num_retries=1)
                    if metadata is not None:
                        if os.path.exists(f'{local_filepath}.part'):
                            os.remove(f'{local_filepath}.part')
                        progress_bar.update(http_request.resumable.size() - progress_last)
                    else:
                        # Write latest progress to part file
                        with open(f'{local_filepath}.part', 'w') as part_fp:
                            json.dump({
                                'resumable_uri': http_request.resumable_uri,
                                'progress_bytes': progress.resumable_progress,
                                'progress_total': progress.total_size,
                                'progress_prc': progress.progress(),
                            }, part_fp, indent=4)
                        # Update progress
                        progress_bar.update(progress.resumable_progress - progress_last)
                        progress_last = progress.resumable_progress

            # Upload finished
            file.uploaded = True
            file.dirty['content'] = False
            file.UpdateMetadata(metadata)
        except pydrive_errors.HttpError as e:
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
        # Create return object
        uploaded_gfile = gfile if is_update and gfile else \
            GDriveFile(pydrive_file=file, gfolder=cloud_folder)
        # Refresh gfolder files cache
        cloud_folder.refresh_files_cache()
        return uploaded_gfile


if __name__ == '__main__':
    GDRIVE_WHICH = 'personal'  # 'personal', 'auth'
    _gd_suffix = '_personal' if GDRIVE_WHICH == 'personal' else ''
    _gc = GDriveCapsule(local_gdrive_root=f'/home/achariso/PycharmProjects/gans-thesis/.gdrive{_gd_suffix}',
                        use_refresh_token=True, update_credentials=True)
    _gfs = GDriveFilesystem(gcapsule=_gc)
    _gf = GDriveFolder.root(_gfs, update_cache=True, cloud_root='12IiDRSnj6r7Jd66Yxz3ZZTn9EFW-Qnqu')
    _gfm = _gf.subfolder_by_name('Models', recursive=False) \
        .subfolder_by_name_or_create('model_name=stylegan_karras', recursive=False) \
        .subfolder_by_name_or_create('Configurations', recursive=False)
    _gfm.upload_file(local_filename='karrasB_z512.yaml', show_progress=True)
