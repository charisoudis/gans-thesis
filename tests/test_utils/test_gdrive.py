import filecmp
import os
import unittest

import torch
from pydrive.drive import GoogleDrive
from torch import nn

from utils.gdrive import get_client_dict, get_gdrive_service, get_root_folders_ids, GDriveModelCheckpoints


class TestGDriveUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.client_secrets_filepath = '/home/achariso/PycharmProjects/gans-thesis/client_secrets.json'
        self.cache_directory = '/home/achariso/PycharmProjects/gans-thesis/.http_cache'
        self.local_chkpts_directory = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints'

        gdrive = get_gdrive_service(client_filepath=self.client_secrets_filepath, cache_root=self.cache_directory)
        self.assertIsNotNone(gdrive)
        self.gdmc = GDriveModelCheckpoints(gdrive=gdrive, local_chkpts_root=self.local_chkpts_directory,
                                           model_name_sep='_')

        self.drive_file_ids_to_remove = []

    def test_get_client_dict(self):
        client_filepaths = ['random.json', self.client_secrets_filepath]
        for fp in client_filepaths:
            if fp == 'random.json':
                with self.assertRaises(FileNotFoundError):
                    get_client_dict(fp)
            else:
                client_dict = get_client_dict(fp)
                for key in ['client_id', 'client_secret', 'refresh_token', 'access_token', 'access_token_expires_at']:
                    self.assertTrue(key in client_dict.keys())

    def test_get_gdrive_service(self):
        client_filepaths = ['random.json', self.client_secrets_filepath]
        for fp in client_filepaths:
            gdrive_or_none = get_gdrive_service(client_filepath=fp, cache_root=None)
            if fp == 'random.json':
                self.assertIsNone(gdrive_or_none)
            else:
                self.assertEqual(GoogleDrive, type(gdrive_or_none))

    def test_get_root_folders_ids(self):
        folder_ids_dict = get_root_folders_ids(gdrive=self.gdmc.gdrive)
        self.assertEqual(dict, type(folder_ids_dict))
        for key in ['GitHub Keys', 'Datasets', 'Model Checkpoints']:
            key = key.lower().replace(' ', '_')
            self.assertTrue(key in folder_ids_dict.keys(), msg=f'{key} not found in folder_ids_dict (keys=' +
                                                               f'{folder_ids_dict.keys()})')

    def test_GDriveModelCheckpoints(self):
        # Check drive scape results
        self.assertTrue('inception' in self.gdmc.chkpt_groups.keys())
        self.assertTrue('pgpg' in self.gdmc.chkpt_groups.keys())
        self.assertTrue('inception' in self.gdmc.latest_chkpts.keys())
        self.assertTrue('pgpg' in self.gdmc.latest_chkpts.keys())

        # Create a new model checkpoint
        test_model = nn.Conv2d(3, 3, 3)
        chkpt_filepath = f'{self.local_chkpts_directory}/test_0000000001_01.pth'
        torch.save({'test': test_model.state_dict()}, chkpt_filepath)

        # Upload the checkpoint file
        self.assertTrue(self.gdmc.upload_model_checkpoint(chkpt_filepath))
        self.assertTrue('test' in self.gdmc.chkpt_groups.keys())
        self.assertTrue('test' in self.gdmc.latest_chkpts.keys())
        self.assertEqual(os.path.basename(chkpt_filepath),
                         self.gdmc.get_latest_model_chkpt_data(model_name='test')['title'])

        self.drive_file_ids_to_remove.append(self.gdmc.latest_chkpts['test']['id'])

        # Rename checkpoint file, since it will be overridden on download
        new_chkpt_filepath = f'{self.local_chkpts_directory}/test_0000000002_01.pth'
        os.rename(chkpt_filepath, new_chkpt_filepath)

        # Download the checkpoint file & compare file contents
        result, dl_chkpt_filepath = self.gdmc.download_model_checkpoint(model_name='test', use_threads=False)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(dl_chkpt_filepath))
        self.assertTrue(filecmp.cmp(new_chkpt_filepath, dl_chkpt_filepath))

    def tearDown(self) -> None:
        # Remove local files created during unit test
        for _root, _, _files in os.walk(self.local_chkpts_directory):
            for _f in _files:
                if _f.startswith('test') and _f.endswith('.pth'):
                    # Remove file
                    os.remove(f'{_root}/{_f}')

        # Remove uploaded files to Google Drive during unit test
        for _id in self.drive_file_ids_to_remove:
            gdmc_file = self.gdmc.gdrive.CreateFile({'id': _id})
            gdmc_file.Delete()

        # Close sockets
        self.gdmc.close()
        del self.gdmc
