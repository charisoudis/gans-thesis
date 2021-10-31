import torch
from IPython.core.display import display
from PIL import Image
from torch import Tensor
from torch.nn import DataParallel
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.deep_fashion import FISBDataset, FISBDataloader
from modules.stylegan import StyleGan
from train_setup import args, run_locally, exec_device, log_level, datasets_groot, models_groot, in_notebook
from utils.dep_free import get_tqdm
from utils.ifaces import FilesystemDataset
from utils.metrics import GanEvaluator

# FIX
torch.cuda.empty_cache()

###################################
# #  Hyper-parameters settings  ###
###################################
#   - training
n_epochs = 300
batch_size = 128 if not run_locally else 4
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
#   - CycleGAN configuration
z_dim = 512
stgan_config_id = f'default_z{z_dim}'  # close to DiscoGAN paper + half precision

###################################
# #   Dataset Initialization    ###
###################################
#   - image transforms:
#     If target_shape is different from load one, resize & crop. If target_shape is different from load shape,
#     convert to grayscale.
#     Update: Now done automatically if you set target_channels, target_shape when instantiating the dataloader.
gen_transforms = FISBDataset.get_image_transforms(target_shape=target_shape, target_channels=target_channels)
#   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
#     > len(dataloader) = <number of batches>
#     > len(dataloader.dataset) = <number of total dataset items>
dataloader = FISBDataloader(dataset_fs_folder_or_root=datasets_groot, batch_size=batch_size, log_level=log_level,
                            image_transforms=gen_transforms, splits=train_test_splits, pin_memory=not run_locally,
                            load_in_memory=not run_locally)
dataset = dataloader.dataset  # save training dataset as `dataset`
#   - ensure dataset is fetched locally and unzipped
if isinstance(dataset, FilesystemDataset):
    dataset.fetch_and_unzip(in_parallel=False, show_progress=True)
elif hasattr(dataset, 'dataset') and isinstance(dataset.dataset, FilesystemDataset):
    dataset.dataset.fetch_and_unzip(in_parallel=False, show_progress=True)
else:
    raise TypeError('dataset must implement utils.ifaces.FilesystemDataset in order to be auto-downloaded and unzipped')
#   - apply rudimentary tests
assert issubclass(dataloader.__class__, DataLoader)
# noinspection PyTypeChecker
assert len(dataloader) == len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)
_real = next(iter(dataloader))
assert tuple(_real.shape) == (batch_size, target_channels, target_shape, target_shape)

###################################
# #    Models Initialization    ###
###################################
#   - initialize evaluator instance (used to run GAN evaluation metrics: FID, IS, PRECISION, RECALL, F1 and SSIM)
evaluator = GanEvaluator(model_fs_folder_or_root=models_groot, gen_dataset=dataset, z_dim=z_dim, device=exec_device,
                         n_samples=metrics_n_samples, batch_size=metrics_batch_size, f1_k=f1_k)
#   - initialize model
chkpt_step = args.chkpt_step
try:
    if chkpt_step == 'latest':
        stgan_chkpt_step = chkpt_step
    elif isinstance(chkpt_step, str) and chkpt_step.isdigit():
        stgan_chkpt_step = int(chkpt_step)
    else:
        stgan_chkpt_step = None
except NameError:
    stgan_chkpt_step = None
# noinspection PyTypeChecker
stgan = StyleGan(model_fs_folder_or_root=models_groot, config_id=stgan_config_id, dataset_len=len(dataset),
                 chkpt_epoch=stgan_chkpt_step, evaluator=evaluator, device=exec_device, log_level=log_level)
########################################################################################################################
# ################################################### DEV LOGGING ######################################################
########################################################################################################################
# stgan._init_gen_disc_opt_scheduler(resolution=64)
########################################################################################################################
stgan.logger.debug(f'Using device: {str(exec_device)}')
stgan.logger.debug(f'Model initialized. Number of params = {stgan.nparams_hr}')
# FIX: Warmup counters before first batch
if stgan.step is None:
    stgan.gforward(batch_size=batch_size)
    stgan.logger.debug(f'Model warmed-up (internal counters).')
# FIX: Dataloader batch_size need update
if stgan.current_batch_size is not None and stgan.current_batch_size != batch_size:
    stgan.logger.debug(f'Updating Dataloader batch_size (from {batch_size} --> {stgan.current_batch_size}).')
    batch_size = stgan.current_batch_size
    dataloader = dataloader.update_batch_size(batch_size=batch_size)
#   - setup multi-GPU training
if torch.cuda.device_count() > 1:
    stgan.gen = DataParallel(stgan.gen)
    stgan.info(f'Using {torch.cuda.device_count()} GPUs for CycleGAN Generator (via torch.nn.DataParallel)')
#   - load dataloader state (from model checkpoint)
if 'dataloader' in stgan.other_state_dicts.keys():
    dataloader.set_state(stgan.other_state_dicts['dataloader'])
    stgan.logger.debug(f'Loaded dataloader state! Current pem_index={dataloader.get_state()["perm_index"]}')

# FIX: Change batch size (if needed)
stgan.update_batch_size(batch_size, sampler_instance=dataloader.sampler)

###################################
# #       Training Loop         ###
###################################
#   - get the correct tqdm instance
exec_tqdm = get_tqdm()
#   - start training loop from last checkpoint's epoch and step
torch.cuda.empty_cache()
gcapture_ready = True
async_results = None
stgan.logger.info(f'[training loop] STARTING (epoch={stgan.epoch}, step={stgan.initial_step})')
for epoch in range(stgan.epoch, n_epochs):
    # Check if the networks should grow
    if stgan.growing() or batch_size != stgan.current_batch_size:
        batch_size = stgan.current_batch_size
        stgan.logger.critical(f'Reinitializing Dataloader... (new batch_size={batch_size})')
        dataloader = dataloader.update_batch_size(batch_size=batch_size)
        stgan.update_batch_size(batch_size, sampler_instance=dataloader.sampler)

    # noinspection PyProtectedMember
    d = {
        'step': stgan.step,
        'initial_step': stgan.initial_step,
        'epoch': stgan.epoch,
        '_counter': stgan._counter,
        'epoch_inc': stgan.epoch_inc,
    }
    # initial_step = stgan.initial_step % len(dataloader)
    stgan.logger.debug('[START OF EPOCH] ' + str(d))

    real: Tensor
    for real in exec_tqdm(dataloader, initial=stgan.initial_step):
        # Downsample images
        if real.shape[-1] != stgan.gen.resolution:
            real = transforms.Resize(size=stgan.gen.resolution, interpolation=Image.BILINEAR)(real)

        # Transfer image batches to GPU
        real = real.to(exec_device)

        # Perform a forward + backward pass + weight update on the Generator & Discriminator models
        disc_loss, gen_loss = stgan(real)

        ################################################################################################################
        # ############################################### DEV LOGGING ##################################################
        ################################################################################################################
        # break
        ################################################################################################################

        # Metrics & Checkpoint Code
        if stgan.step % checkpoint_step == 0:
            # Check if another upload is pending
            if not gcapture_ready and async_results:
                # Wait for previous upload to finish
                stgan.logger.warning('Waiting for previous gcapture() to finish...')
                [r.wait() for r in async_results]
                stgan.logger.warning('DONE! Starting new capture now.')
            # Capture current model state, including metrics and visualizations
            async_results = stgan.gcapture(checkpoint=True, metrics=stgan.step % metrics_step == 0, visualizations=True,
                                           dataloader=dataloader, in_parallel=True, show_progress=True,
                                           delete_after=True)
        # Visualization code
        elif stgan.step % display_step == 0:
            visualization_img = stgan.visualize()
            visualization_img.show() if not in_notebook() else display(visualization_img)

        # Check if a pending checkpoint upload has finished
        if async_results:
            gcapture_ready = all([r.ready() for r in async_results])
            if gcapture_ready:
                stgan.logger.info(f'gcapture() finished')
                if stgan.latest_checkpoint_had_metrics:
                    stgan.logger.info(str(stgan.latest_metrics))
                async_results = None

        # If run locally one pass is enough
        if run_locally and gcapture_ready:
            break

    # If run locally one pass is enough
    if run_locally:
        break

    # noinspection PyProtectedMember
    d = {
        'step': stgan.step,
        'initial_step': stgan.initial_step,
        'epoch': stgan.epoch,
        '_counter': stgan._counter,
        'epoch_inc': stgan.epoch_inc,
    }
    stgan.logger.debug('[END OF EPOCH] ' + str(d))

# Check if a pending checkpoint exists
if async_results:
    ([r.wait() for r in async_results])
    stgan.logger.info(f'last gcapture() finished')
    if stgan.latest_checkpoint_had_metrics:
        stgan.logger.info(str(stgan.latest_metrics))
    async_results = None

# Training finished!
stgan.logger.info('[training loop] DONE')

########################################################################################################################
# ################################################### DEV LOGGING ######################################################
########################################################################################################################
# img = stgan.visualize()
# img.show()
# img = stgan.visualize(reproducible=True)
# img.show()
########################################################################################################################
