from typing import Union

from utils.filesystems.local import LocalCapsule, LocalFile, LocalFolder, LocalFilesystem


class ColabCapsule(LocalCapsule):
    """
    ColabCapsule Class:
    This is class is used to initiate the connection to GoogleDrive API and eventually open a TCP socket via which we
    will submit HTTP request for the various needed operations (e.g. download/upload/list ops).
    """

    def __init__(self, local_gdrive_root: str = '/content/drive/MyDrive', project_root: str = '/'):
        """
        ColabCapsule class constructor.
        :param (str) local_gdrive_root: absolute path to the local directory where Google Drive files will be synced to
        :param (str) project_root: relative path to folder considered as root for the project
        """
        super(ColabCapsule, self).__init__(local_root=f'{local_gdrive_root}{project_root}'.rstrip('/'))


class ColabFile(LocalFile):
    """
    ColabFile Class:
    This class, implementing `FilesystemFile` interface, is used to download/list info of files stored in Google Drive.
    """

    def __init__(self, filename: str, cfolder: 'ColabFolder'):
        """
        ColabFile class constructor.
        :param (str) filename: basename of file in local fs
        :param (ColabFolder) cfolder: an `utils.gdrive.ColabFolder` instance with the folder info inside of which lives
                                      this files
        """
        super(ColabFile, self).__init__(filename=filename, cfolder=cfolder)


class ColabFolder(LocalFolder):
    """
    ColabFolder Class:
    This class, implementing `FilesystemFolder` interface, is used to transfer files from/to respective Google Drive
    folder.
    """

    @staticmethod
    def root(capsule_or_fs: Union[ColabCapsule, 'ColabFilesystem']) -> 'ColabFolder':
        fs = capsule_or_fs if isinstance(capsule_or_fs, ColabFilesystem) else ColabFilesystem(ccapsule=capsule_or_fs)
        return ColabFolder(fs=fs, local_root=fs.local_root, parent=None)


class ColabFilesystem(LocalFilesystem):
    """
    ColabFilesystem Class:
    This class is used to interact with files stored in the locally-mounted Google Drive via native OS calls.
    """
