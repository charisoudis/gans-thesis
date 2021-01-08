from typing import Union

from utils.filesystems.local import LocalCapsule, LocalFile, LocalFolder, LocalFilesystem


class KaggleCapsule(LocalCapsule):
    """
    KaggleCapsule Class:
    This is class is used to initiate the connection to GoogleDrive API and eventually open a TCP socket via which we
    will submit HTTP request for the various needed operations (e.g. download/upload/list ops).
    """

    def __init__(self, local_root: str = '/kaggle/working/GoogleDrive'):
        """
        KaggleCapsule class constructor.
        :param (str) local_root: absolute path to the local directory where Google Drive files will be synced to
        """
        super(KaggleCapsule, self).__init__(local_root=local_root)


class KaggleFile(LocalFile):
    """
    KaggleFile Class:
    This class, implementing `FilesystemFile` interface, is used to download/list info of files stored in Google Drive.
    """

    def __init__(self, filename: str, cfolder: 'KaggleFolder'):
        """
        KaggleFile class constructor.
        :param (str) filename: basename of file in local fs
        :param (KaggleFolder) cfolder: an `utils.gdrive.KaggleFolder` instance with the folder info inside of which
                                       exists this file
        """
        super(KaggleFile, self).__init__(filename=filename, cfolder=cfolder)


class KaggleFolder(LocalFolder):
    """
    KaggleFolder Class:
    This class, implementing `FilesystemFolder` interface, is used to transfer files from/to respective Google Drive \
    folder.
    """

    @staticmethod
    def root(capsule_or_fs: Union[KaggleCapsule, 'KaggleFilesystem']) -> 'KaggleFolder':
        fs = capsule_or_fs if isinstance(capsule_or_fs, KaggleFilesystem) else KaggleFilesystem(ccapsule=capsule_or_fs)
        return KaggleFolder(fs=fs, local_root=fs.local_root, parent=None)


class KaggleFilesystem(LocalFilesystem):
    """
    KaggleFilesystem Class:
    This class is used to interact with files stored in the locally-mounted Google Drive via native OS calls.
    """
