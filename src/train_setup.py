import argparse
import os
import sys

import torch
from IPython import get_ipython

from utils.data import ManualSeedReproducible
from utils.dep_free import in_notebook
from utils.filesystems.gdrive.colab import ColabFilesystem, ColabFolder, ColabCapsule
from utils.filesystems.gdrive.remote import GDriveCapsule, GDriveFilesystem, GDriveFolder
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
# Flag to run first test batches locally
from utils.plot import ensure_matplotlib_fonts_exist

##########################################
# #         Parse CLI Arguments        ###
##########################################
parser = argparse.ArgumentParser(description='Trains GAN model in PyTorch.')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='execution device (\'cpu\', or \'cuda\')')
parser.add_argument('--log_level', type=str, default='debug', choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help='default log level (\'debug\', \'info\', \'warning\', \'error\' or \'critical\')')
parser.add_argument('--chkpt_step', type=str, default='latest',
                    help='model checkpoint to be loaded (\'latest\' or str or int)')
parser.add_argument('--seed', type=int, default=42,
                    help='random generators seed value (default: 42)')
parser.add_argument('-use_refresh_token', action='store_true',
                    help='if set will use client_secrets.json to connect to Google Drive, else will ask for auth code')
parser.add_argument('--run_locally', action='store_true',
                    help='flag must be present to start local running (aka first pass run)')
# New GDrive root (e.g. "/Education/AUTH/COURSES/10th Semester - Thesis/ThesisGStorage")
parser.add_argument('--gdrive_new_root', type=str, default='/',
                    help='Relative path of Google Drive folder to be considered as root')
args = parser.parse_args()

##########################################
# #     Environment Initialization     ###
##########################################
run_locally = True
if in_notebook() and not args.run_locally:
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
        run_locally = False
        local_gdrive_root = '/workspace/GoogleDrive'  # vast.ai
        if not os.path.exists(local_gdrive_root):
            local_gdrive_root = input('local_gdrive_root = ')
assert os.path.exists(local_gdrive_root), f'local_gdrive_root={local_gdrive_root} NOT FOUND'
os.environ['TRAIN_EXEC_ENV'] = exec_env

# Check if GPU is available
exec_device = torch.device('cuda:0' if 'cuda' == args.device and torch.cuda.is_available() else 'cpu')
os.environ['TRAIN_EXEC_DEV'] = str(exec_device)

# Get log level
log_level = args.log_level
os.environ['TRAIN_LOG_LEVEL'] = log_level

# Reproducibility
seed = ManualSeedReproducible.manual_seed(args.seed)

##########################################
# #  GDrive Filesystem Initialization  ###
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
    use_refresh_token = args.use_refresh_token
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
