import unittest

from pydrive.drive import GoogleDrive

from utils.gdrive import get_client_dict, get_gdrive_service, get_root_folders_ids, GDriveModelCheckpoints


class TestGDriveUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.client_secrets_filepath = '/home/achariso/PycharmProjects/gans-thesis/client_secrets.json'
        self.cache_directory = '/home/achariso/PycharmProjects/gans-thesis/.http_cache'
        self.local_chkpts_directory = '/home/achariso/PycharmProjects/gans-thesis/.checkpoints'

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
        gdrive = get_gdrive_service(client_filepath=self.client_secrets_filepath, cache_root=self.cache_directory)
        self.assertIsNotNone(gdrive)
        folder_ids_dict = get_root_folders_ids(gdrive=gdrive)
        self.assertEqual(dict, type(folder_ids_dict))
        for key in ['GitHub Keys', 'Datasets', 'Model Checkpoints']:
            key = key.lower().replace(' ', '_')
            self.assertTrue(key in folder_ids_dict.keys(), msg=f'{key} not found in folder_ids_dict (keys=' +
                                                               f'{folder_ids_dict.keys()})')

    def test_GDriveModelCheckpoints(self):
        gdrive = get_gdrive_service(client_filepath=self.client_secrets_filepath, cache_root=self.cache_directory)
        self.assertIsNotNone(gdrive)

        gdmc = GDriveModelCheckpoints(gdrive=gdrive, local_chkpts_root=self.local_chkpts_directory, model_name_sep='_')
        self.assertListEqual(['inception', 'pgpg'], sorted(gdmc.chkpt_groups.keys()))
        self.assertListEqual(['inception', 'pgpg'], sorted(gdmc.latest_chkpts.keys()))
        # TODO: finish this unit test

