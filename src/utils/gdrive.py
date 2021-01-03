import atexit
import json
import os
from datetime import datetime as dt, timedelta
from multiprocessing.pool import ApplyResult, ThreadPool
from typing import Optional, List, Tuple, Union, Dict

import httplib2
# noinspection PyProtectedMember
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
from oauth2client.client import OAuth2Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import ApiRequestError, FileNotUploadedError, FileNotDownloadableError, GoogleDriveFile
from pydrive.settings import InvalidConfigError
from requests import post as post_request

from utils.command_line_logger import CommandLineLogger
from utils.dep_free import get_tqdm
from utils.ifaces import CloudCapsule, CloudFilesystem, CloudFolder


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
            _service = build('drive', 'v2', http=_http)
            return _service, _http, ocredentials
        except KeyError or ValueError or InvalidConfigError:
            return None, None, None


class GDriveFilesystem(CloudFilesystem):
    """
    GDriveFilesystem Class:
    This class is used to interact with files stored in Google Drive via the `google-api-client` python lib.
    """

    def __init__(self, gcapsule: GDriveCapsule):
        """
        GDriveFilesystem class constructor.
        :param (GDriveCapsule) gcapsule: a `utils.gdrive.GDriveCapsule` instance to interact with Google Drive cloud
                                         filesystem
        """
        self.tqdm = get_tqdm()
        self.logger = CommandLineLogger(log_level='info', name=self.__class__.__name__)
        # Create a thread pool for parallel uploads/downloads
        self.thread_pool = ThreadPool(processes=1)
        atexit.register(self.thread_pool.close)
        # Check given args
        # TODO
        # Save args
        self.gcapsule = gcapsule
        self.gservice_files = gcapsule.gservice.files()
        self.local_root = gcapsule.local_root

    def download_file(self, cloud_file: GoogleDriveFile, local_filepath: str, in_parallel: bool = False,
                      show_progress: bool = False) -> Union[ApplyResult, bool]:
        # If use threads, start a new thread to curry out the download and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=self.download_file_thread,
                                                        args=(self, cloud_file, local_filepath))
            return async_result

        # If file exists and .download file not, this means the cloud file has already been downloaded and, as so, we
        # can safely return True
        dl_file_path = f'{local_filepath}.download'
        if os.path.exists(local_filepath) and not os.path.exists(dl_file_path):
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
            return True
        except ApiRequestError or FileNotUploadedError or FileNotDownloadableError as e:
            self.logger.critical(f'[GDriveFilesystem::download_file] {str(e)}')
            return False

    def list_files(self, cloud_folder: GoogleDriveFile) -> List[GoogleDriveFile]:
        gdrive_query = f"'{cloud_folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' " + \
                       f"and trashed=false"
        # return [{
        #     'id': _file_data['id'],
        #     'title': _file_data['title'],
        #     'size': humanize.naturalsize(int(_file_data['fileSize'])) if 'fileSize' in _file_data.keys() else '-'
        # } for _file_data in self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()]
        return self.gcapsule.pydrive.ListFile({'q': gdrive_query}).GetList()

    def upload_local_file(self, local_filepath: str, cloud_folder: GoogleDriveFile, delete_after: bool = False,
                          in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, Optional[GoogleDriveFile]]:
        # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
        # caller if wanted
        if in_parallel:
            async_result = self.thread_pool.apply_async(func=GDriveFilesystem.upload_local_file_thread,
                                                        args=(self, local_filepath, cloud_folder, delete_after))
            return async_result

        # Find model name from file path
        file_basename = os.path.basename(local_filepath)
        # Create GoogleDrive API new File instance
        file = self.gcapsule.pydrive.CreateFile({'title': file_basename, 'parents': [{'id': cloud_folder['id']}]})
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
        return file


class GDriveFolder(CloudFolder):
    """
    GDriveFolder Class:
    This class, implementing `CloudFolder` interface, is used to transfer files from/to respective Google Drive folder.
    """

    ExcludedFolders = ['Colab Notebooks']

    def __init__(self, fs: GDriveFilesystem, cloud_root: GoogleDriveFile, local_root: str):
        """
        GDriveFolder class constructor.
        :param (GDriveFilesystem) fs: a `utils.gdrive.GDriveFilesystem` instance to interact with GoogleDrive filesystem
        :param (dict) cloud_root: the data of the Google Drive root folder
        :param (str) local_root: the absolute path of the local root folder
        """
        self.fs = fs
        # Check args
        if not os.path.exists(local_root) or not os.path.isdir(local_root):
            self.fs.logger.warning(f'local_root={local_root}: NOT FOUND/Readable. Creating dir now...')
            assert 0 == os.system(f'mkdir -p "{local_root}"')
        # Save args
        self.cloud_root = cloud_root
        self.local_root = local_root
        # Init dict (enable json serializing of class instances)
        dict.__init__(self, cloud_root=cloud_root, local_root=local_root)

    def download(self, cloud_file: GoogleDriveFile, in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, bool]:
        return self.fs.download_file(cloud_file=cloud_file, local_filepath=f'{self.local_root}/{cloud_file["title"]}',
                                     in_parallel=in_parallel, show_progress=show_progress)

    def download_by_name(self, filename: str, in_parallel: bool = False, show_progress: bool = False) \
            -> Union[ApplyResult, bool]:
        """
        Calls `GDriveFolder::download` if finds a file in file list whose name matches the given :attr:`filename`.
        :param filename: the file name to search for
        :param in_parallel: see `GDriveFolder::download`
        :param show_progress: see `GDriveFolder::download`
        :return: see `GDriveFolder::download`
        """
        for file in self.list():
            if filename == file['title']:
                return self.download(cloud_file=file, in_parallel=in_parallel, show_progress=show_progress)

        raise FileNotFoundError(f'"{filename}" NOT FOUND in Google Drive (under "{self.cloud_root["title"]}")')

    def list(self) -> List[GoogleDriveFile]:
        return self.fs.list_files(cloud_folder=self.cloud_root)

    def upload(self, local_filename: str, delete_after: bool = False, in_parallel: bool = False,
               show_progress: bool = False) -> Union[ApplyResult, Optional[GoogleDriveFile]]:
        return self.fs.upload_local_file(local_filepath=f'{self.local_root}/{os.path.basename(local_filename)}',
                                         cloud_folder=self.cloud_root, delete_after=delete_after,
                                         in_parallel=in_parallel, show_progress=show_progress)

    @staticmethod
    def instance(fs: GDriveFilesystem, cloud_root: GoogleDriveFile, folder_name: str) -> 'GDriveFolder':
        return GDriveFolder(fs=fs, cloud_root=cloud_root, local_root=f'{fs.local_root}/{folder_name}')


def get_gdrive_root_folders(gcapsule: GDriveCapsule) -> Dict[str, GDriveFolder]:
    """
    Get all folder under Google Drive's root as `GDriveFolder` objects that allow interaction with the respective cloud
    folders.
    :param (GDriveCapsule) gcapsule: the `utils.gdrive.GDriveCapsule` instance to "talk" to GoogleDrive API
    :return: a dict of the form {"<folder_name_slugged>": <GDriveFolder instance>}
    """
    # Initialize filesystem instance
    fs = GDriveFilesystem(gcapsule=gcapsule)
    # Create GoogleDrive API query to fetch folders under Google Drive root
    get_folders_under_root_query = "'root' in parents and mimeType = 'application/vnd.google-apps.folder' " + \
                                   "and trashed=false"
    return_dict = {}
    for _f in gcapsule.pydrive.ListFile({'q': get_folders_under_root_query}).GetList():
        folder_name = _f['title']
        if folder_name in GDriveFolder.ExcludedFolders:
            continue

        return_dict[folder_name.lower().replace(' ', '_')] = GDriveFolder.instance(fs=fs, cloud_root=_f,
                                                                                   folder_name=folder_name)
    return return_dict


# class GDriveModelCheckpoints(object):
#
#     def __init__(self, gdrive: GoogleDrive, local_chkpts_root: Optional[str] = None, model_name_sep: str = '_'):
#         """
#         GDriveModelCheckpoints class constructor.
#         :param gdrive: the pydrive.drive.GoogleDrive instance to access GoogleDrive API
#         :param (optional) local_chkpts_root: absolute path to model checkpoints directory
#         :param model_name_sep: separator of checkpoint file names (to retrieve model name and group accordingly)
#         """
#         self.tqdm = get_tqdm()
#         self.logger = CommandLineLogger(log_level='info')
#
#         assert gdrive is not None, "gdrive attribute is None"
#         self.gdrive = gdrive
#         self.gservice = gdrive.auth.service
#
#         # Local Checkpoints root
#         # Check if running inside Colab or Kaggle (auto prefixing)
#         if 'google.colab' in sys.modules or 'google.colab' in str(get_ipython()) or 'COLAB_GPU' in os.environ:
#             local_chkpts_root = f'/content/drive/MyDrive/Model Checkpoints'
#         elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
#             local_chkpts_root = f'/kaggle/working/Model Checkpoints'
#         elif not local_chkpts_root:
#             local_chkpts_root: str = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints'
#         assert os.path.exists(local_chkpts_root) and os.path.isdir(local_chkpts_root), 'Checkpoints dir not existent ' \
#                                                                                        'or not readable'
#         self.local_chkpts_root = local_chkpts_root
#
#         self.folder_id = get_root_folders_ids(gdrive=gdrive)['model_checkpoints']
#         self.file_ids = self.__class__.get_model_checkpoint_files_ids(gdrive, chkpts_folder_id=self.folder_id)
#
#         # Get checkpoint files data and register latest checkpoints for each model found in GDrive's checkpoints folder
#         self.model_name_sep = model_name_sep
#         self.chkpt_groups = group_by_prefix(self.file_ids, dict_key='title', separator=model_name_sep)
#         self.latest_chkpts = {}
#         for model_name in self.chkpt_groups.keys():
#             self.latest_chkpts[model_name] = self._get_latest_model_chkpt_data(model_name=model_name)
#
#     def _get_latest_model_chkpt_data(self, model_name: str) -> dict:
#         """
#         Get the latest checkpoint file data from the list of file checkpoints for the given :attr:`model_name`.
#         :param model_name: the model_name (should appear in :attr:`self.latest_chkpts` keys)
#         :return: a dict containing the latest checkpoint file data (e.g. {"id": <fileId>, "title": <fileTitle>, ...})
#         """
#         # Check if model checkpoint exists in latest checkpoints dict
#         if model_name in self.latest_chkpts.keys():
#             return self.latest_chkpts[model_name]
#         # Else, find the latest checkpoint, save in dict and return
#         model_chkpts = self.chkpt_groups[model_name]
#         latest_chkpt = sorted(model_chkpts, key=lambda _f: _f['title'], reverse=True)[0]
#         self.latest_chkpts[model_name] = latest_chkpt
#         return latest_chkpt
#
#     def get_model_chkpt_data(self, model_name: str, step: Union[str, int] = 'latest') -> dict:
#         """
#         :param model_name: the model_name (should appear in :attr:`self.chkpt_groups` keys)
#         :param step: either an int of the step the model checkpoint created at or the string 'latest' to get GoogleDrive
#                      info of the latest model checkpoint ("x-steps"=="x batches passed through model")
#         :return: a dict containing the checkpoint file data (e.g. {"id": <fileId>, "title": <fileTitle>, ...}) for the
#                  specified step
#         """
#         if model_name not in self.chkpt_groups.keys():
#             raise AttributeError(f'model_name="{model_name}" not found in chkpt_groups keys:' +
#                                  f' {str(self.chkpt_groups.keys())}')
#
#         if type(step) == str and 'latest' == 'step':
#             return self._get_latest_model_chkpt_data(model_name=model_name)
#
#         print(self.chkpt_groups)
#
#     @staticmethod
#     def download_checkpoint_thread(gdmc: 'GDriveModelCheckpoints', chkpt_data: dict) -> None:
#         result, _ = gdmc.download_checkpoint(chkpt_data=chkpt_data, use_threads=False)
#         if not result:
#             raise ValueError('gdmc.download_checkpoint() returned False')
#
#     def download_checkpoint(self, chkpt_data: dict, use_threads: bool = False, show_progress: bool = False) \
#             -> Tuple[Union[Thread, bool], Optional[str]]:
#         """
#         Download model checkpoint described in :attr:`chkpt_data` from Google Drive to local filesystem at
#         :attr:`self.local_chkpts_root`.
#         :param chkpt_data: a dict describing the gdrive data of checkpoint (e.g.
#                            {'id': <Google Drive FileID>, 'title': <Google Drive FileTitle>, ...}
#         :param use_threads: set to True to have download carried out be a new Thread, thus returning to caller
#                             immediately with the Thread instance
#         :param show_progress: set to True to have an auto-updating Tqdm progress bar to indicate download progress
#         :return: a tuple containing either a threading.Thread instance (if :attr:`use_threads`=True) or a boolean set to
#                  True if no error occurred or False in different case and the local filepath as a str object
#         """
#         # Get destination file path
#         local_chkpt_filepath = f'{self.local_chkpts_root}/{chkpt_data["title"]}'
#         # If use threads, start a new thread to curry out the download and return the thread object to be joined by the
#         # caller if wanted
#         if use_threads:
#             thread = Thread(target=GDriveModelCheckpoints.download_checkpoint_thread, args=(self, chkpt_data))
#             thread.start()
#             return thread, local_chkpt_filepath
#
#         # If file exists, do not overwrite, just return
#         if os.path.exists(local_chkpt_filepath):
#             return True, local_chkpt_filepath
#         # Download file from Google Drive to local checkpoints root
#         try:
#             # file = self.gdrive.CreateFile({'id': chkpt_data['id']})
#             # local_file_handler = io.BytesIO()
#             file_request = self.gservice.files().get_media(fileId=chkpt_data['id'])
#             # Check if previous partial download exists, and read current progress
#             dl_file_path = f'{local_chkpt_filepath}.download'
#             if os.path.exists(dl_file_path) and os.path.exists(local_chkpt_filepath):
#                 with open(dl_file_path, 'wb') as dl_fp:
#                     _progress = dl_fp.read().splitlines()[0]
#                     if _progress and str == type(_progress) and _progress.isdigit():
#                         _progress = int(_progress)
#                     else:
#                         _progress = 0
#             else:
#                 _progress = 0
#             # Open file pointer and begin downloading/writing bytes from Google Drive
#             with open(local_chkpt_filepath, 'ab' if os.path.exists(local_chkpt_filepath) else 'wb') as fp:
#                 file_downloader = MediaIoBaseDownload(fp, file_request)
#                 file_downloader._progress = _progress
#                 with self.tqdm(disable=not show_progress, total=100) as progress_bar:
#                     done = False
#                     while not done:
#                         mdp, done = file_downloader.next_chunk()
#                         # Write next (first) byte in a .download file to be able to resume file download
#                         with open(dl_file_path, 'w') as dl_fp:
#                             dl_fp.write(mdp.resumable_progress)
#                         progress_bar.moveto(int(mdp.progress() * 100))
#             # file.GetContentFile(filename=local_chkpt_filepath, callback=progress_bar.update)
#             return True, local_chkpt_filepath
#         except ApiRequestError or FileNotUploadedError or FileNotDownloadableError as e:
#             self.logger.critical(f'download_checkpoint() FAILed: {str(e)}')
#             return False, None
#
#     def download_model_checkpoint(self, model_name: str, step: Union[str, int] = 'latest', use_threads: bool = False) \
#             -> Tuple[Union[Thread, bool], Optional[str]]:
#         """
#         Download latest model checkpoint from Google Drive to local filesystem at :attr:`self.local_chkpts_root`.
#         :param model_name: model' name (it is also the checkpoint file name prefix)
#         :param step: either an int of the step the model checkpoint created at or the string 'latest' to get GoogleDrive
#                      info of the latest model checkpoint ("x-steps"=="x batches passed through model")
#         :param use_threads: set to True to have download carried out be a new Thread, thus returning to caller
#                             immediately with the Thread instance
#         :return: a tuple containing either a threading.Thread instance (if :attr:`use_threads`=True) or a boolean set to
#                  True if no error occurred or False in different case and the local filepath as a str object
#         """
#         model_chkpt_data = self.get_model_chkpt_data(model_name=model_name, step=step)
#         return self.download_checkpoint(chkpt_data=model_chkpt_data, use_threads=use_threads)
#
#     @staticmethod
#     def upload_model_checkpoint_thread(gdmc: 'GDriveModelCheckpoints', filepath: str, delete_after: bool) -> None:
#         if not gdmc.upload_model_checkpoint(chkpt_filepath=filepath, use_threads=False, delete_after=delete_after):
#             raise ValueError('gdmc.upload_model_checkpoint() returned False')
#
#     def upload_model_checkpoint(self, chkpt_filepath: str, use_threads: bool = False,
#                                 delete_after: bool = False) -> Union[Thread, bool]:
#         """
#         Upload locally-saved model checkpoint to Google Drive for permanent storage.
#         :param chkpt_filepath: absolute path of the locally-saved checkpoint file
#         :param use_threads: set to True to run upload function in a separate thread, thus returning immediately
#                             to caller
#         :param delete_after: set to True to have the local file deleted after successful upload
#         :return: a Thread object is use_threads was set else a bool object set to True if upload completed successfully,
#                  False with corresponding messages otherwise
#         """
#         # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
#         # caller if wanted
#         if use_threads:
#             thread = Thread(target=GDriveModelCheckpoints.upload_model_checkpoint_thread,
#                             args=(self, chkpt_filepath, delete_after))
#             thread.start()
#             return thread
#
#         # Find model name from file path
#         chkpt_file_basename = os.path.basename(chkpt_filepath)
#         model_name = chkpt_file_basename.split(self.model_name_sep, maxsplit=1)[0]
#         # Create GoogleDrive API new File instance
#         file = self.gdrive.CreateFile({'title': chkpt_file_basename, 'parents': [{'id': self.folder_id}]})
#         # Load local file data into the File instance
#         assert os.path.exists(chkpt_filepath)
#         file.SetContentFile(chkpt_filepath)
#         try:
#             file.Upload()
#         except ApiRequestError as e:
#             self.logger.critical(f'File upload failed (chkpt_filepath={chkpt_filepath}): {str(e)}')
#             return False
#         # Close file handle
#         file.content.close()
#         # Register uploaded checkpoint in class
#         try:
#             file_size = file['fileSize']
#         except KeyError or FileNotUploadedError as e:
#             self.logger.critical(f'Could not fetch file size: {str(e)}')
#             file_size = '-'
#         new_file_data = {'id': file['id'], 'title': file['title'], 'size': file_size}
#         self.latest_chkpts[model_name] = new_file_data
#         self.chkpt_groups.setdefault(model_name, [])
#         self.chkpt_groups[model_name].append(new_file_data)
#         # Delete file after successful upload
#         if delete_after:
#             try:
#                 os.remove(chkpt_filepath)
#             except FileNotFoundError or AttributeError as e:
#                 self.logger.critical(f'Could not remove chkpt_filepath (at {chkpt_filepath}): {str(e)}')
#                 return False
#         return True
#
#     def close(self) -> None:
#         self.gdrive.auth.http.close()
#
#     def __del__(self) -> None:
#         self.close()
#
#     @staticmethod
#     def get_model_checkpoint_files_ids(gdrive: GoogleDrive, chkpts_folder_id: Optional[str] = None) -> List[dict]:
#         """
#         Get a list of all model checkpoint files data from given Google Drive instance.
#         :param gdrive: an pydrive.drive.GoogleDrive instance
#         :param (optional) chkpts_folder_id: the Google Drive fileId attribute  of the Model Checkpoints folder or None
#                                             to fetch automatically from Google Drive instance
#         :return: a list of dictionaries, of the form
#                  [{"id": "<file_id>", "title": "<file_title>", "size": "<human_readable_file_size>"}, ...]
#         """
#         if not chkpts_folder_id:
#             chkpts_folder_id = get_root_folders_ids(gdrive=gdrive)['model_checkpoints']
#
#         _return_ids = []
#         for _file_data in gdrive.ListFile({'q': f"'{chkpts_folder_id}' in parents and " +
#                                                 "mimeType != 'application/vnd.google-apps.folder' and " +
#                                                 "trashed=false"
#                                            }).GetList():
#             _return_ids.append({
#                 'id': _file_data['id'],
#                 'title': _file_data['title'],
#                 'size': humanize.naturalsize(int(_file_data['fileSize'])) if 'fileSize' in _file_data.keys() else '-'
#             })
#         return _return_ids
#
#     @staticmethod
#     def instance(client_secrets_filepath: str = '/home/achariso/PycharmProjects/gans-thesis/client_secrets.json',
#                  cache_directory: Optional[str] = '/home/achariso/PycharmProjects/gans-thesis/.http_cache') \
#             -> Optional['GDriveModelCheckpoints']:
#         """
#         Get a new GDriveModelCheckpoints instance while also instantiating firstly a GoogleDrive API service
#         :param client_secrets_filepath: absolute path to client_secrets.json
#         :param cache_directory: HTTP cache directory or None to disable cache
#         :return: a new self instance or None in case of failure connecting to GoogleDrive API service
#         """
#         # When running locally, disable OAuthLib's HTTPs verification
#         if client_secrets_filepath.startswith('/home/achariso'):
#             os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
#
#         # Instantiate GoogleDrive API service
#         gdrive = get_gdrive_service(client_filepath=client_secrets_filepath, cache_root=cache_directory)
#         if not gdrive:
#             return None
#
#         # Instantiate self
#         return GDriveModelCheckpoints(gdrive=gdrive)

if __name__ == '__main__':
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _c = GDriveCapsule(local_gdrive_root=_local_gdrive_root, use_http_cache=True, update_credentials=True)
    _gf = get_gdrive_root_folders(gcapsule=_c)
    # print(json.dumps(_gf, indent=4))

    metrics_gfolder = _gf['model_checkpoints']
    print(metrics_gfolder.download_by_name(filename='inception_v3_google-1a9a5a14.pth', in_parallel=False,
                                           show_progress=True))
