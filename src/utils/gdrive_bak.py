import os
from multiprocessing.pool import ApplyResult
from typing import Optional, Tuple, Union

from pydrive.files import GoogleDriveFile

from utils.command_line_logger import CommandLineLogger
from utils.string import group_by_prefix


# noinspection DuplicatedCode
class GDriveModelCheckpoints(object):

    def __init__(self, chkpts_gfolder, model_name_sep: str = '_'):
        """
        GDriveModelCheckpoints class constructor.
        :param (GDriveFolder) chkpts_gfolder: a `utils.gdrive.GDriveFolder` instance to interact with dataset folder
                                               in Google Drive
        :param model_name_sep: separator of checkpoint file names (to retrieve model name and group accordingly)
        """
        self.logger = CommandLineLogger(log_level=os.getenv('TRAIN_LOG_LEVEL', 'info'))
        # Save args
        self.gfolder = chkpts_gfolder
        self.local_chkpts_root = chkpts_gfolder.local_root
        # Get checkpoint files data and register latest checkpoints for each model found in GDrive's checkpoints folder
        self.model_name_sep = model_name_sep
        self.chkpt_groups = group_by_prefix(chkpts_gfolder.files, dict_key='title', separator=model_name_sep)
        self.latest_chkpts = {}
        for model_name in self.chkpt_groups.keys():
            self.latest_chkpts[model_name] = self._get_latest_model_chkpt(model_name=model_name)

    def _get_latest_model_chkpt(self, model_name: str) -> GoogleDriveFile:
        """
        Get the latest checkpoint file data from the list of file checkpoints for the given :attr:`model_name`.
        :param model_name: the model_name (should appear in :attr:`self.latest_chkpts` keys)
        :return: a `pydrive.files.GoogleDriveFile` object containing the latest checkpoint file GoogleDrive API info
                 (e.g. {"id": <fileId>, "title": <fileTitle>, ...})
        """
        # Check if model checkpoint exists in latest checkpoints dict
        if model_name in self.latest_chkpts.keys():
            return self.latest_chkpts[model_name]
        # Else, find the latest checkpoint, save in dict and return
        model_chkpts = self.chkpt_groups[model_name]
        latest_chkpt = sorted(model_chkpts, key=lambda _f: _f['title'], reverse=True)[0]
        self.latest_chkpts[model_name] = latest_chkpt
        return latest_chkpt

    def get_model_chkpt(self, model_name: str, step: Union[str, int] = 'latest') -> Optional[GoogleDriveFile]:
        """
        Get the checkpoint file that corresponds to the given :attr:`step`.
        :param model_name: the model_name (should appear in :attr:`self.chkpt_groups` keys)
        :param step: either an int of the step the model checkpoint created at or the string 'latest' to get GoogleDrive
                     info of the latest model checkpoint ("x-steps"=="x batches passed through model")
        :return: a `pydrive.files.GoogleDriveFile` object containing the checkpoint file data
                 (e.g. {"id": <fileId>, "title": <fileTitle>, ...}) for the specified step
        """
        # Check args
        if model_name not in self.chkpt_groups.keys():
            raise AttributeError(f'model_name="{model_name}" not found in chkpt_groups keys:' +
                                 f' {str(self.chkpt_groups.keys())}')
        # If latest checkpoint requested, return from the saved latest checkpoint files in self.latest_chkpts
        if type(step) == str and 'latest' == 'step':
            return self._get_latest_model_chkpt(model_name=model_name)
        # Search for model checkpoint at given step in self.chkpt_groups
        for _f in self.chkpt_groups[model_name]:
            if _f.startswith(f'{model_name}{self.model_name_sep}{str(step).zfill(10)}'):
                return _f
        return None

    def download_checkpoint(self, chkpt_gfile: GoogleDriveFile, in_parallel: bool = False,
                            show_progress: bool = False) -> Tuple[Union[ApplyResult, bool], Optional[str]]:
        """
        Download model checkpoint described in :attr:`chkpt_data` from Google Drive to local filesystem at
        :attr:`self.local_chkpts_root`.
        :param chkpt_gfile: a `pydrive.filed.GoogleDriveFile` object describing the gdrive data of checkpoint (e.g.
                            {'id': <Google Drive FileID>, 'title': <Google Drive FileTitle>, ...}
        :param in_parallel: set to True to have download carried out be a new Thread, thus returning to caller
                            immediately with the Thread instance
        :param show_progress: set to True to have an auto-updating Tqdm progress bar to indicate download progress
        :return: a tuple containing either a `multiprocessing.pool.ApplyResult` instance (if :attr:`in_parallel`=True)
                 or a `bool` set to True if no error occurred or False in different case and the local filepath as an
                `str` object
        """
        # Get destination file path
        local_chkpt_filepath = f'{self.local_chkpts_root}/{chkpt_gfile["title"]}'
        # Download file using underlying GDriveFolder instance
        return self.gfolder.download(cloud_file=chkpt_gfile, in_parallel=in_parallel,
                                     show_progress=show_progress), local_chkpt_filepath

    def download_model_checkpoint(self, model_name: str, step: Union[str, int] = 'latest', in_parallel: bool = False,
                                  show_progress: bool = False) -> Tuple[Union[ApplyResult, bool], Optional[str]]:
        """
        Download latest model checkpoint from Google Drive to local filesystem at :attr:`self.local_chkpts_root`.
        :param model_name: model' name (it is also the checkpoint file name prefix)
        :param step: either an int of the step the model checkpoint created at or the string 'latest' to get GoogleDrive
                     info of the latest model checkpoint ("x-steps"=="x batches passed through model")
        :param in_parallel: set to True to have download carried out be a new Thread, thus returning to caller
                            immediately with the Thread instance
        :param show_progress: set to True to have an auto-updating Tqdm progress bar to indicate download progress
        :return: a tuple containing either a `multiprocessing.pool.ApplyResult` object (if :attr:`in_parallel`=True)
                 or a `bool` set to True if no error occurred or False in different case and the local filepath as an
                `str` object
        """
        model_chkpt = self.get_model_chkpt(model_name=model_name, step=step)
        return self.download_checkpoint(chkpt_gfile=model_chkpt, in_parallel=in_parallel, show_progress=show_progress)

    def upload_model_checkpoint(self, chkpt_filename: str, in_parallel: bool = False,
                                delete_after: bool = False) -> Union[ApplyResult, bool]:
        """
        Upload locally-saved model checkpoint to Google Drive for permanent storage.
        :param chkpt_filename: file name (NOT absolute path) of the locally-saved checkpoint file
        :param in_parallel: set to True to run upload function in a separate thread, thus returning immediately
                            to caller
        :param delete_after: set to True to have the local file deleted after successful upload
        :return: a `multiprocessing.pool.ApplyResult` object is :attr:`in_parallel` was set else a `bool` set to `True`
                 if upload completed successfully, `False` with corresponding messages otherwise
        """
        return self.gfolder.upload(local_filename=chkpt_filename, in_parallel=in_parallel, delete_after=delete_after)
