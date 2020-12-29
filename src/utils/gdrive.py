import json
import os
from datetime import datetime as dt, timedelta
from threading import Thread
from typing import Union, Optional, List

import httplib2
import humanize
# noinspection PyPackageRequirements
from googleapiclient.discovery import build
from oauth2client.client import OAuth2Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import ApiRequestError, FileNotUploadedError
from pydrive.settings import InvalidConfigError
from requests import post as post_request

from utils.command_line_logger import CommandLineLogger
from utils.string import group_by_prefix


def get_client_dict(client_filepath: str) -> dict:
    """
    Get updated client OAuth data, ready for instantiating google.oauth2.credentials.Credentials.
    Info on generating OAuth credentials can be found here:
    https://www.ibm.com/support/knowledgecenter/SS6KM6/com.ibm.appconnect.dev.doc/how-to-guides-for-apps/getting-oauth-credentials-for-google-applications.html
    :param client_filepath: absolute path to the JSON file from Google developers console
    :return: a dict object with the initial data plus the access_token, the expires_at ISO string and possibly a
             renewed refresh_token
    """
    with open(client_filepath) as _fp:
        _client_dict = json.load(_fp)
        if 'web' in _client_dict.keys():
            _client_dict = _client_dict['web']

    should_refresh = 'access_token' not in _client_dict.keys() or \
                     'access_token_expires_at' not in _client_dict.keys() or \
                     (dt.utcnow() - dt.fromisoformat(_client_dict['access_token_expires_at'])).seconds < 300
    if not should_refresh:
        return _client_dict

    response = post_request(_client_dict['token_uri'], {
        'grant_type': 'refresh_token',
        'client_id': _client_dict['client_id'],
        'client_secret': _client_dict['client_secret'],
        'refresh_token': _client_dict['refresh_token']
    }).json()
    print(response)
    _client_dict['access_token'] = response['access_token']
    expires_in = int(response['expires_in'])
    _client_dict['access_token_expires_at'] = (dt.utcnow() + timedelta(seconds=expires_in)).isoformat()
    _client_dict['refresh_token'] = response['refresh_token'] if 'refresh_token' in response.keys() \
        else _client_dict['refresh_token']

    # Write back to json file
    with open(client_filepath, 'w') as fp:
        json.dump(_client_dict, fp, indent=4)

    return _client_dict


def get_gdrive_service(client_filepath: str, cache_root: Optional[str] = None) -> Union[GoogleDrive, None]:
    """
    Get an authenticated pydrive.drive.GoogleDrive instance ready for use as a cloud disk.
    :param client_filepath: the filepath to the client_secrets.json file
    :param (optional) cache_root: absolute path to cache cache directory or None to disable cache
    :return: a pydrive.drive.GoogleDrive instance or None if any exception occurs
    """
    credentials = get_client_dict(client_filepath=client_filepath)
    credentials = {**credentials, **{
        'scopes': ['https://www.googleapis.com/auth/drive'],
        'token_expiry': dt.fromisoformat(credentials['access_token_expires_at']),
        'user_agent': None
    }}
    ocredentials = OAuth2Credentials(**{x: credentials[x] for x in credentials.keys() if x in [
        'access_token', 'client_id', 'client_secret', 'refresh_token',
        'token_expiry', 'token_uri', 'user_agent', 'revoke_uri',
        'id_token', 'token_response', 'scopes', 'token_info_uri', 'id_token_jwt'
    ]})

    try:
        _http = ocredentials.authorize(httplib2.Http(cache=cache_root))
        _service = build('drive', 'v2', http=_http)

        gauth = GoogleAuth()
        gauth.settings['client_config_file'] = client_filepath
        gauth.credentials = ocredentials
        gauth.http = _http
        gauth.service = _service

        return GoogleDrive(auth=gauth)
    except KeyError or ValueError or InvalidConfigError:
        return None


def get_root_folders_ids(gdrive: GoogleDrive) -> dict:
    """
    Get the ids of all folder under GDrive's root.
    :param gdrive: the pydrive.drive.GoogleDrive instance
    :return: a dict of the form {"<folder_title_slugged>": "<folder_id>", ...}
    """
    _return_ids = {}
    for _folder_data in gdrive.ListFile({'q': f"'root' in parents and " +
                                              "mimeType = 'application/vnd.google-apps.folder' and " +
                                              "trashed=false"
                                         }).GetList():
        _folder_id = _folder_data['id']
        _folder_title = _folder_data['title']
        _return_ids[_folder_title.lower().replace(' ', '_')] = _folder_id
    return _return_ids


class GDriveModelCheckpoints(object):

    def __init__(self, gdrive: GoogleDrive, model_name_sep: str = '_'):
        """
        GDriveModelCheckpoints class constructor.
        :param gdrive: the pydrive.drive.GoogleDrive instance to access GoogleDrive API
        :param model_name_sep: separator of checkpoint file names (to retrieve model name and group accordingly)
        """
        self.logger = CommandLineLogger(log_level='info')

        self.gdrive = gdrive
        self.folder_id = get_root_folders_ids(gdrive=gdrive)['model_checkpoints']
        self.file_ids = self.__class__.get_model_checkpoint_files_ids(gdrive, chkpts_folder_id=self.folder_id)

        # Get checkpoint files data and register latest checkpoints for each model found in GDrive's checkpoints folder
        self.model_name_sep = model_name_sep
        self.chkpt_groups = group_by_prefix(self.file_ids, dict_key='title', separator=model_name_sep)
        self.latest_chkpts = {}
        for model_name in self.chkpt_groups.keys():
            self.latest_chkpts[model_name] = self.get_latest_model_chkpt(model_name=model_name)

    def get_latest_model_chkpt(self, model_name: str) -> dict:
        """
        Get the latest checkpoint file data from the list of file checkpoints for the given :attr:`model_name`.
        :param model_name: the model_name (should appear in :attr:`self.chkpt_groups` keys)
        :return: a dict containing the latest checkpoint file data (e.g. fileId and title)
        """
        if model_name in self.latest_chkpts.keys():
            return self.latest_chkpts[model_name]

        if model_name not in self.chkpt_groups.keys():
            raise AttributeError(f'model_name="{model_name}" not found in chkpt_groups keys:' +
                                 f' {str(self.chkpt_groups.keys())}')

        model_chkpts = self.chkpt_groups[model_name]
        return sorted(model_chkpts, key=lambda _f: _f['title'], reverse=True)[0]

    def upload_model_checkpoint(self, chkpt_filepath: str, use_threads: bool = True,
                                delete_after: bool = False) -> Union[Thread, bool]:
        """
        Upload locally-saved model checkpoint to Google Drive for permanent storage.
        :param chkpt_filepath: absolute path of the locally-saved checkpoint file
        :param use_threads: set to True to run upload function in a separate thread, thus returning immediately
                            to caller
        :param delete_after: set to True to have the local file deleted after successful upload
        :return: a Thread object is use_threads was set else a bool object set to True if upload completed successfully,
                 False with corresponding messages otherwise
        """
        # If use threads, start a new thread to curry out the upload and return the thread object to be joined by the
        # caller if wanted
        if use_threads:
            thread = Thread(target=GDriveModelCheckpoints.upload_model_checkpoint_thread,
                            args=(self, chkpt_filepath, delete_after))
            thread.start()
            return thread

        # Find model name from file path
        chkpt_file_basename = os.path.basename(chkpt_filepath)
        model_name = chkpt_file_basename.split(self.model_name_sep, maxsplit=1)[0]
        # Create GoogleDrive API new File instance
        file = self.gdrive.CreateFile({'title': chkpt_file_basename, 'parents': [{'id': self.folder_id}]})
        # Load local file data into the File instance
        file.SetContentFile(chkpt_filepath)
        try:
            file.Upload()
        except ApiRequestError as e:
            self.logger.critical(f'File upload failed (chkpt_filepath={chkpt_filepath}): {str(e)}')
            return False
        # Register uploaded checkpoint in class
        try:
            file_size = file['fileSize']
        except KeyError or FileNotUploadedError as e:
            self.logger.critical(f'Could not fetch file size: {str(e)}')
            file_size = '-'
        new_file_data = {'id': file['id'], 'title': file['title'], 'size': file_size}
        self.latest_chkpts[model_name] = new_file_data
        self.chkpt_groups.setdefault(model_name, [])
        self.chkpt_groups[model_name].append(new_file_data)
        # Delete file after successful upload
        if delete_after:
            try:
                os.remove(chkpt_filepath)
            except FileNotFoundError or AttributeError as e:
                self.logger.critical(f'Could not remove chkpt_filepath (at {chkpt_filepath}): {str(e)}')
                return False
        return True

    @staticmethod
    def get_model_checkpoint_files_ids(gdrive: GoogleDrive, chkpts_folder_id: Optional[str] = None) -> List[dict]:
        """
        Get a list of all model checkpoint files data from given Google Drive instance.
        :param gdrive: an pydrive.drive.GoogleDrive instance
        :param (optional) chkpts_folder_id: the Google Drive fileId attribute  of the Model Checkpoints folder or None
                                            to fetch automatically from Google Drive instance
        :return: a list of dictionaries, of the form
                 [{"id": "<file_id>", "title": "<file_title>", "size": "<human_readable_file_size>"}, ...]
        """
        if not chkpts_folder_id:
            chkpts_folder_id = get_root_folders_ids(gdrive=gdrive)['model_checkpoints']

        _return_ids = []
        for _file_data in gdrive.ListFile({'q': f"'{chkpts_folder_id}' in parents and " +
                                                "mimeType != 'application/vnd.google-apps.folder' and " +
                                                "trashed=false"
                                           }).GetList():
            _return_ids.append({
                'id': _file_data['id'],
                'title': _file_data['title'],
                'size': humanize.naturalsize(int(_file_data['fileSize'])) if 'fileSize' in _file_data.keys() else '-'
            })
        return _return_ids

    @staticmethod
    def upload_model_checkpoint_thread(gdmc, filepath: str, delete_after: bool) -> None:
        if gdmc.upload_model_checkpoint(chkpt_filepath=filepath, use_threads=False, delete_after=delete_after):
            print('success!!!!!!!!!!!!!!!!')


if __name__ == '__main__':
    # When running locally, disable OAuthLib's HTTPs verification
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    client_secrets_filepath = '/home/achariso/PycharmProjects/gans-thesis/client_secrets.json'
    cache_directory = '/home/achariso/PycharmProjects/gans-thesis/.http_cache'
    _gdrive = get_gdrive_service(client_filepath=client_secrets_filepath, cache_root=cache_directory)

    _gdmc = GDriveModelCheckpoints(_gdrive)
    print('before')
    _thread = _gdmc.upload_model_checkpoint(
        chkpt_filepath=client_secrets_filepath,
        use_threads=True,
        delete_after=False
    )
    print('after upload_model_checkpoint()')
    # print(json.dumps(_gdmc.chkpt_groups, indent=4))
    # print(json.dumps(_gdmc.latest_chkpts, indent=4))
    _thread.join()
    print('[DONE]')
