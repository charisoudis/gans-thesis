import os
import sys

import torch
from IPython import get_ipython
from IPython.core.display import display
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from datasets.deep_fashion import ICRBDataset, ICRBCrossPoseDataloader
from modules.pgpg import PGPG
from utils.dep_free import get_tqdm, in_notebook
from utils.filesystems.gdrive.colab import ColabFilesystem, ColabFolder, ColabCapsule
from utils.filesystems.gdrive.remote import GDriveCapsule, GDriveFilesystem, GDriveFolder
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.ifaces import FilesystemDataset
from utils.metrics import GanEvaluator
# Flag to run first test batches locally
from utils.plot import ensure_matplotlib_fonts_exist

run_locally = True
if in_notebook():
    run_locally = False  # local runs are performed vis IDE runs (and thus terminal)

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

# Check if GPU is available
exec_device = torch.device('cuda' if torch.cuda.is_available() and not run_locally else 'cpu')

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
print(f'log_level={log_level}')

###################################
###  Hyper-parameters settings  ###
###################################
#   - training
n_epochs = 100
batch_size = 48 if not run_locally else 2
train_test_splits = [90, 10]  # for a 90% training - 10% evaluation set split
#   - evaluation
metrics_n_samples = 1000 if not run_locally else 2
metrics_batch_size = 32 if not run_locally else 1
f1_k = 3 if not run_locally else 1
#   - visualizations / checkpoints steps
display_step = 200
checkpoint_step = 600
metrics_step = 1800  # evaluate model every 3 checkpoints
#   - dataset
target_shape = 128
target_channels = 3
skip_pose_norm = True
#   - PGPG config file
pgpg_config_id = f'{target_shape}_MSE_256_6_4_5_none_none_1e4_true_false_false'  # as proposed in the original paper

##########################################
###  GDrive Filesystem Initialization  ###
##########################################
#   - define FilesystemFolder to interact with files/folders under the root folder on Google Drive
if exec_env == 'colab':
    # Colab filesystem is a locally-mounted filesystem. Interacts with native OS calls.
    fs = ColabFilesystem(ccapsule=ColabCapsule())
    groot = ColabFolder.root(capsule_or_fs=fs)
elif run_locally and False:
    # Local filesystem (basically one directory under given root). Interacts with native OS calls.
    fs = LocalFilesystem(ccapsule=LocalCapsule(local_root=local_gdrive_root))
    groot = LocalFolder.root(capsule_or_fs=fs)
else:
    # Remove filesystem. Interacts via GoogleDrive API calls.
    gcapsule = GDriveCapsule(local_gdrive_root=local_gdrive_root, use_http_cache=True, update_credentials=True,
                             use_refresh_token=run_locally)
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

###################################
###   Dataset Initialization    ###
###################################
#   - image transforms:
#     If target_shape is different from load one, resize & crop. If target_shape is different from load shape,
#     convert to grayscale.
#     Update: Now done automatically if you set target_channels, target_shape when instantiating the dataloader.
gen_transforms = ICRBDataset.get_image_transforms(target_shape=target_shape, target_channels=target_channels)
#   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
#     > len(dataloader) = <number of batches>
#     > len(dataloader.dataset) = <number of total dataset items>
dataloader = ICRBCrossPoseDataloader(dataset_fs_folder_or_root=datasets_groot, batch_size=batch_size,
                                     image_transforms=gen_transforms, skip_pose_norm=skip_pose_norm,
                                     splits=train_test_splits, pin_memory=not run_locally, log_level=log_level)
dataset = dataloader.dataset
#   - ensure dataset is fetched locally and unzipped
if isinstance(dataset, FilesystemDataset):
    dataset.fetch_and_unzip(in_parallel=False, show_progress=True)
elif hasattr(dataset, 'dataset') and isinstance(dataset.dataset, FilesystemDataset):
    dataset.dataset.fetch_and_unzip(in_parallel=False, show_progress=True)
else:
    raise TypeError('dataset must implement utils.ifaces.FilesystemDataset in order to be auto-downloaded and unzipped')
#   - apply basic tests
assert issubclass(dataloader.__class__, DataLoader)
assert len(dataloader) == len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)
_image_1, _image_2, _dense_pose_2 = next(iter(dataloader))
assert tuple(_image_1.shape) == (batch_size, target_channels, target_shape, target_shape)
assert tuple(_image_2.shape) == (batch_size, target_channels, target_shape, target_shape)
assert tuple(_dense_pose_2.shape) == (batch_size, target_channels, target_shape, target_shape)

###################################
###    Models Initialization    ###
###################################
#   - initialize evaluator instance (used to run GAN evaluation metrics: FID, IS, PRECISION, RECALL, F1 and SSIM)
evaluator = GanEvaluator(model_fs_folder_or_root=models_groot, gen_dataset=dataset, target_index=1, device=exec_device,
                         condition_indices=(0, 2), n_samples=metrics_n_samples, batch_size=metrics_batch_size,
                         f1_k=f1_k)
#   - initialize model
global chkpt_step
try:
    if chkpt_step == 'latest':
        pgpg_chkpt_step = chkpt_step
    elif isinstance(chkpt_step, str) and chkpt_step.isdigit():
        pgpg_chkpt_step = int(chkpt_step)
    else:
        pgpg_chkpt_step = None
except NameError:
    pgpg_chkpt_step = None
pgpg = PGPG(model_fs_folder_or_root=models_groot, config_id=pgpg_config_id, dataset_len=len(dataset),
            chkpt_epoch=pgpg_chkpt_step, evaluator=evaluator, device=exec_device, log_level=log_level)
pgpg.logger.debug(f'Model initialized. Number of params = {pgpg.nparams_hr}')
#   - load dataloader state (from model checkpoint)
if 'dataloader' in pgpg.other_state_dicts.keys():
    dataloader.set_state(pgpg.other_state_dicts['dataloader'])
    pgpg.logger.debug(f'Loaded dataloader state! Current pem_index={dataloader.get_state()["perm_index"]}')

###################################
###       Training Loop         ###
###################################
#   - get the correct tqdm instance
exec_tqdm = get_tqdm()
#   - start training loop from last checkpoint's epoch and step
gcapture_ready = True
async_results = None
pgpg.logger.info(f'[training loop] STARTING (epoch={pgpg.epoch}, step={pgpg.initial_step})')
for epoch in range(pgpg.epoch, n_epochs):
    image_1: Tensor
    image_2: Tensor
    pose_2: Tensor
    for image_1, image_2, pose_2 in get_tqdm()(dataloader, initial=pgpg.initial_step):
        # Transfer image batches to GPU
        image_1 = image_1.to(exec_device)
        image_2 = image_2.to(exec_device)
        pose_2 = pose_2.to(exec_device)

        # Perform a forward + backward pass + weight update on the Generator & Discriminator models
        disc_loss, gen_loss = pgpg(image_1=image_1, image_2=image_2, pose_2=pose_2)

        # Metrics & Checkpoint Code
        if pgpg.step % checkpoint_step == 0:
            # Check if another upload is pending
            if not gcapture_ready and async_results:
                # Wait for previous upload to finish
                pgpg.logger.warning('Waiting for previous gcapture() to finish...')
                [r.wait() for r in async_results]
                pgpg.logger.warning('DONE! Starting new capture now.')
            # Capture current model state, including metrics and visualizations
            async_results = pgpg.gcapture(checkpoint=True, metrics=pgpg.step % metrics_step == 0, visualizations=True,
                                          dataloader=dataloader, in_parallel=True, show_progress=run_locally,
                                          delete_after=False)
        # Visualization code
        elif pgpg.step % display_step == 0:
            visualization_img = pgpg.visualize()
            visualization_img.show() if not in_notebook() else display(visualization_img)

        # Check if a pending checkpoint upload has finished
        if async_results:
            gcapture_ready = all([r.ready() for r in async_results])
            if gcapture_ready:
                pgpg.logger.info(f'gcapture() finished')
                if pgpg.latest_checkpoint_had_metrics:
                    pgpg.logger.info(str(pgpg.latest_metrics))
                async_results = None

        # If run locally one pass is enough
        if run_locally and gcapture_ready:
            break

    # If run locally one pass is enough
    if run_locally:
        break

# Check if a pending checkpoint exists
if async_results:
    ([r.wait() for r in async_results])
    pgpg.logger.info(f'last gcapture() finished')
    if pgpg.latest_checkpoint_had_metrics:
        pgpg.logger.info(str(pgpg.latest_metrics))
    async_results = None

# Training finished!
pgpg.logger.info('[training loop] DONE')
