import os
import sys

import torch
from IPython import get_ipython

from utils.dep_free import in_notebook
from utils.filesystems.gdrive.colab import ColabFilesystem, ColabFolder, ColabCapsule
from utils.filesystems.gdrive.remote import GDriveCapsule, GDriveFilesystem, GDriveFolder
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
# Flag to run first test batches locally
from utils.plot import ensure_matplotlib_fonts_exist

##########################################
###     Environment Initialization     ###
##########################################
run_locally = True
if in_notebook():
    run_locally = False  # local runs are performed vis IDE runs (and thus terminal)
os.environ['TRAIN_ENV'] = 'local' if run_locally else 'nonlocal'

# Check if running inside Colab or Kaggle
if 'google.colab' in sys.modules or 'google.colab' in str(get_ipython()) or 'COLAB_GPU' in os.environ:
    exec_env = 'colab'
    local_gdrive_root = '/content/drive/MyDrive'
    run_locally = False
elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    exec_env = 'kaggle'
    local_gdrive_root = '/kaggle/working/GoogleDrive'
    run_locally = False
else:
    exec_env = 'ssh'
    local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    if not os.path.exists(local_gdrive_root):
        local_gdrive_root = input('local_gdrive_root = ')
        run_locally = False
assert os.path.exists(local_gdrive_root), f'local_gdrive_root={local_gdrive_root} NOT FOUND'
os.environ['TRAIN_EXEC_ENV'] = exec_env

# Check if GPU is available
exec_device = torch.device('cuda:0' if torch.cuda.is_available() and not run_locally else 'cpu')
os.environ['TRAIN_EXEC_DEV'] = str(exec_device)

# Get log level
global log_level
if in_notebook():
    try:
        log_level = f'{log_level}'
    except NameError:
        print('You should define log_level variable before running this script')
        log_level = input('log_level=')
        assert log_level in ['debug', 'info', 'warning', 'error']
else:
    log_level = 'info' if not run_locally else 'debug'
os.environ['TRAIN_LOG_LEVEL'] = log_level
print(f'log_level={log_level}')

##########################################
###  GDrive Filesystem Initialization  ###
##########################################
#   - define FilesystemFolder to interact with files/folders under the root folder on Google Drive
if exec_env == 'colab':
    # Colab filesystem is a locally-mounted filesystem. Interacts with native OS calls.
    fs = ColabFilesystem(ccapsule=ColabCapsule())
    groot = ColabFolder.root(capsule_or_fs=fs)
elif run_locally:
    # Local filesystem (basically one directory under given root). Interacts with native OS calls.
    fs = LocalFilesystem(ccapsule=LocalCapsule(local_root=local_gdrive_root))
    groot = LocalFolder.root(capsule_or_fs=fs)
else:
    # Remote filesystem. Interacts via GoogleDrive API calls.
    global use_refresh_token
    try:
        use_refresh_token = use_refresh_token or False
    except NameError:
        use_refresh_token = run_locally
    gcapsule = GDriveCapsule(local_gdrive_root=local_gdrive_root, use_http_cache=True, update_credentials=True,
                             use_refresh_token=use_refresh_token)
    fs = GDriveFilesystem(gcapsule=gcapsule)
    groot = GDriveFolder.root(capsule_or_fs=fs, update_cache=True)
#   - define immediate sub-folders of root folder
# print(json.dumps(groot.subfolders, indent=4))
datasets_groot = groot.subfolder_by_name('Datasets')
models_groot = groot.subfolder_by_name('Models')
fonts_groot = groot.subfolder_by_name('Fonts')
#   - ensure that system and matplotlib fonts directories exist and have the correct font files
rebuilt_fonts = ensure_matplotlib_fonts_exist(fonts_groot, force_rebuild=False)
if rebuilt_fonts and exec_env != 'ssh':
    groot.fs.logger.critical('Fonts rebuilt! Terminating python process now.')
    os.kill(os.getpid(), 9)
