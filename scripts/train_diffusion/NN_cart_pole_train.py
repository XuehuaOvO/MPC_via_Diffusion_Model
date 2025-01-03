import numpy as np
import os
import torch
import torch.nn as nn
import shutil
from tqdm.autonotebook import tqdm
import copy
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import dict_to_device, DEFAULT_TENSOR_ARGS, to_numpy
from collections import defaultdict
from mpd.trainer import get_specified_dataset, get_dataset # , get_model, get_loss, get_summary
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

########## Setting ############
STATE_DIM = 5
BATCH_SIZE = 512
HORIZON = 32

DATASET_SUBDIR = 'CartPole-NMPC' # 'diff_mpc_2024/5_6_noise_3'

MODEL_SAVED_DIRECTORY = 'logs/nn_cart_test1' # under the train_diffusion folder
EXTRA_MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/nn_trained_models/nn_cart_pole_test1'

# training data range (for get_specified_dataset)
# NORMAL_POS_RANGE = range(0,16000)
# NORMAL_NEG_RANGE = range(16000,32000)
# NOISY_POS_RANGE = range(32000,352000)
# NOISY_NEG_RANGE = range(352000,672000)

# NN Class
class AMPCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(AMPCNet, self).__init__()
        # Define the hidden layers and output layer
        self.hidden1 = nn.Linear(input_size, 2)  # First hidden layer with 2 neurons
        self.hidden2 = nn.Linear(2, 50)          # Second hidden layer with 50 neurons
        self.hidden3 = nn.Linear(50, 50)         # Third hidden layer with 50 neurons
        self.output = nn.Linear(50, output_size) # Output layer

    def forward(self, x, batch_size, horizon):
        # Forward pass through the network with the specified activations
        x = torch.tanh(self.hidden1(x))          # Tanh activation for first hidden layer
        x = torch.tanh(self.hidden2(x))          # Tanh activation for second hidden layer
        x = torch.tanh(self.hidden3(x))          # Tanh activation for third hidden layer
        x = self.output(x)                       # Linear activation (no activation function) for the output layer

        # reshape the output
        x = x.view(batch_size, horizon, 1) # 512(batch size)*8*1

        return x

class EMA:
    """
    https://github.com/jannerm/diffuser
    (empirical) exponential moving average parameters
    """

    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def do_summary(
        summary_fn,
        train_steps_current,
        model,
        batch_dict,
        loss_info,
        datasubset,
        **kwargs
):
    if summary_fn is None:
        return

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        summary_fn(train_steps_current,
                   model,
                   batch_dict=batch_dict,
                   loss_info=loss_info,
                   datasubset=datasubset,
                   **kwargs
                   )

    # set model to training mode
    model.train()

def save_nn_models_to_disk(models_prefix_l, epoch, total_steps, checkpoints_dir=None):
    for model, prefix in models_prefix_l:
        if model is not None:
            save_model_to_disk(model, epoch, total_steps, checkpoints_dir, prefix=f'{prefix}_')
            # for submodule_key, submodule_value in model.submodules.items():                   # no submodule in nn
            #     save_model_to_disk(submodule_value, epoch, total_steps, checkpoints_dir,
            #                        prefix=f'{prefix}_{submodule_key}_')
                
def save_model_to_disk(model, epoch, total_steps, checkpoints_dir=None, prefix='model_'):
    # If the model is frozen we do not save it again, since the parameters did not change
    if hasattr(model, 'is_frozen') and model.is_frozen:
        return

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}current_state_dict.pth'))
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}current.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}.pth'))

def save_losses_to_disk(train_losses, val_losses, checkpoints_dir=None):
    np.save(os.path.join(checkpoints_dir, f'train_losses.npy'), train_losses)
    np.save(os.path.join(checkpoints_dir, f'val_losses.npy'), val_losses)

# fixed seed
seed = 0
fix_random_seed(seed)

# model input & output size
input_size = STATE_DIM    # Define your input size based on your problem
output_size = HORIZON    # Define your output size based on your problem (e.g., regression or single class prediction)
model = AMPCNet(input_size, output_size)


# @single_experiment_yaml
# def experiment(
########################################################################
# Dataset
dataset_subdir = DATASET_SUBDIR
include_velocity = False

########################################################################

# Training parameters
batch_size = BATCH_SIZE
lr = 3e-3
# num_train_steps = 5000

use_ema = True
use_amp = False

# model saving address
model_saving_address =  EXTRA_MODEL_PATH

# Summary parameters
steps_til_summary = 2000
# summary_class = 'SummaryTrajectoryGeneration'

# steps_til_ckpt = 10000

########################################################################
device = 'cuda'

debug = True

########################################################################
# MANDATORY

results_dir = MODEL_SAVED_DIRECTORY
os.makedirs(results_dir, exist_ok=True)
ema_decay = 0.995
step_start_ema = 1000 
update_ema_every = 10
summary_fn=None
steps_per_validation=10
clip_grad=False
clip_grad_max_norm=1.0
max_steps=None
steps_til_checkpoint=100000
train_losses_info = {}
val_loss_info = {}

########################################################################

# device
device = get_torch_device(device=device)
print(f'device --{device}')
tensor_args = {'device': device, 'dtype': torch.float32}

# move model to device
model = model.to(device)

# Dataset (get_dataset  get_specified_dataset)

# train_subset, train_dataloader, val_subset, val_dataloader = get_specified_dataset(
#     dataset_class='InputsDataset',
#     include_velocity=include_velocity,
#     dataset_subdir=dataset_subdir,
#     batch_size=batch_size,
#     results_dir=results_dir,
#     save_indices=True,
#     normal_pos_range = NORMAL_POS_RANGE,
#     normal_neg_range = NORMAL_NEG_RANGE,
#     noisy_pos_range = NOISY_POS_RANGE,
#     noisy_neg_range = NOISY_NEG_RANGE,
#     tensor_args=tensor_args
# )

train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
    dataset_class='InputsDataset',
    include_velocity=include_velocity,
    dataset_subdir=dataset_subdir,
    batch_size=batch_size,
    results_dir=results_dir,
    save_indices=True,
    tensor_args=tensor_args
)

# Loss
loss_fn = val_loss_fn = criterion = torch.nn.MSELoss()

model.train()

# Number of epochs (how many times to loop over the entire dataset)
epochs = 500

train_steps_current = 0

model_dir=results_dir

print(f'\n------- TRAINING STARTED -------\n')
print("Current CUDA device:", torch.cuda.current_device())
print(f"epochs {epochs}")
print(f'model_dir -- {model_dir}')
print(f'lr -- {lr}')
print(f'model_saving_address -- {model_saving_address}')

ema_model = None
if use_ema:
    # Exponential moving average model
    ema = EMA(beta=ema_decay)
    ema_model = copy.deepcopy(model)

# Model optimizers
optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

# Automatic Mixed Precision
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if val_dataloader is not None:
    assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

## Build saving directories
os.makedirs(model_dir, exist_ok=True)

summaries_dir = os.path.join(model_dir, 'summaries')
os.makedirs(summaries_dir, exist_ok=True)

checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)


stop_training = False
train_steps_current = 0

# save models before training
save_nn_models_to_disk([(model, 'model'), (ema_model, 'ema_model')], 0, 0, checkpoints_dir)


with tqdm(total=(len(train_dataloader)-1) * epochs, mininterval=1 if debug else 60) as pbar:
    train_losses_l = []
    validation_losses_l = []
    for epoch in range(epochs):
        model.train()  # set model to training mode
        for step, train_batch_dict in enumerate(train_dataloader):
            ###### drop the last batch (incomplete size) #####
            if step == len(train_dataloader) - 1:
                break

            ####################################################################################################
            # TRAINING LOSS
            ####################################################################################################
            with TimerCUDA() as t_training_loss:
                train_batch_dict = dict_to_device(train_batch_dict, tensor_args['device'])

                # derive the inputs and targets
                inputs = train_batch_dict['condition_normalized']
                # print(f'inputs -- {inputs}')
                targets = train_batch_dict['inputs_normalized']
                # print(f'targets  -- {targets}')

                # Forward pass: Get predictions from the model
                outputs = model(inputs,batch_size,horizon = HORIZON)

                # Compute the loss
                loss = criterion(outputs, targets)

                train_loss_batch = 0.
                train_losses_log = {}
                # for loss_name, loss in loss.item():
                single_loss = loss.mean()
                train_loss_batch += single_loss
                    # train_losses_log

            ####################################################################################################
            # SUMMARY
            if train_steps_current % steps_til_summary == 0:
                # TRAINING
                print(f"\n-----------------------------------------")
                print(f"train_steps_current: {train_steps_current}")
                print(f"t_training_loss: {t_training_loss.elapsed:.4f} sec")
                print(f"Total training loss {train_loss_batch:.4f}")
                print(f"Training losses {loss}")

                train_losses_l.append((train_steps_current, train_losses_log))

                with TimerCUDA() as t_training_summary:
                    do_summary(
                        summary_fn,
                        train_steps_current,
                        ema_model if ema_model is not None else model,
                        train_batch_dict,
                        train_losses_info,
                        train_subset,
                        prefix='TRAINING ',
                        debug=debug,
                        tensor_args=tensor_args
                    )
                print(f"t_training_summary: {t_training_summary.elapsed:.4f} sec")

                ################################################################################################
                # VALIDATION LOSS and SUMMARY
                validation_losses_log = {}
                if val_dataloader is not None:
                    with TimerCUDA() as t_validation_loss:
                        print("Running validation...")
                        val_losses = defaultdict(list)
                        total_val_loss = 0.
                        for step_val, batch_dict_val in enumerate(val_dataloader):
                            if step_val == len(val_dataloader) - 1:
                                break
                            batch_dict_val = dict_to_device(batch_dict_val, tensor_args['device'])

                            # derive the inputs and targets
                            inputs_val = batch_dict_val['condition_normalized']
                            print(f'inputs_val size -- {inputs_val.size()}')
                            targets_val = batch_dict_val['inputs_normalized']
                            print(f'targets_val size -- {targets_val.size()}')

                            # Forward pass: Get predictions from the model
                            outputs_val = model(inputs_val,batch_size, horizon = HORIZON)

                            # Compute the loss
                            loss_val = criterion(outputs_val, targets_val)

                            name = 'NN Loss'
                            single_loss = to_numpy(loss_val)
                            val_losses[name].append(single_loss)
                            total_val_loss += np.mean(single_loss).item()

                            if step_val == steps_per_validation:
                                break

                        validation_losses = {}
                        for loss_name, loss in val_losses.items():
                            single_loss_val = np.mean(loss).item()
                            validation_losses[f'VALIDATION {loss_name}'] = single_loss_val
                            print("... finished validation.")

                    print(f"t_validation_loss: {t_validation_loss.elapsed:.4f} sec")
                    print(f"Validation losses {validation_losses}")

                    validation_losses_log = validation_losses
                    validation_losses_l.append((train_steps_current, validation_losses_log))

                    # The validation summary is done only on one batch of the validation data
                    with TimerCUDA() as t_validation_summary:
                        do_summary(
                            summary_fn,
                            train_steps_current,
                            ema_model if ema_model is not None else model,
                            batch_dict_val,
                            val_loss_info,
                            val_subset,
                            prefix='VALIDATION ',
                            debug=debug,
                            tensor_args=tensor_args
                        )
                    print(f"t_valididation_summary: {t_validation_summary.elapsed:.4f} sec")

                # wandb.log({**train_losses_log, **validation_losses_log}, step=train_steps_current)

            ####################################################################################################
            # Early stopping
            # if early_stopper.early_stop(total_val_loss):
            #     print(f'Early stopped training at {train_steps_current} steps.')
            #     stop_training = True

            ####################################################################################################
            # OPTIMIZE TRAIN LOSS BATCH
            with TimerCUDA() as t_training_optimization:
                for optim in optimizers:
                    optim.zero_grad()

                scaler.scale(train_loss_batch).backward()

                if clip_grad:
                    for optim in optimizers:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=clip_grad_max_norm if isinstance(clip_grad, bool) else clip_grad
                    )

                for optim in optimizers:
                    scaler.step(optim)

                scaler.update()

                if ema_model is not None:
                    if train_steps_current % update_ema_every == 0:
                        # update ema
                        if train_steps_current < step_start_ema:
                            # reset parameters ema
                            ema_model.load_state_dict(model.state_dict())
                        ema.update_model_average(ema_model, model)

            if train_steps_current % steps_til_summary == 0:
                print(f"t_training_optimization: {t_training_optimization.elapsed:.4f} sec")

            ####################################################################################################
            # SAVING
            ####################################################################################################
            pbar.update(1)
            train_steps_current += 1

            if (steps_til_checkpoint is not None) and (train_steps_current % steps_til_checkpoint == 0):
                save_nn_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                                    epoch, train_steps_current, checkpoints_dir)
                save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)
                print(f"\n-----------------------------------------")
                saved_main_folder = model_saving_address
                os.makedirs(saved_main_folder, exist_ok=True)
                middle_model_dir = os.path.join(saved_main_folder, str(train_steps_current))
                # os.makedirs(middle_model_dir, exist_ok=True)
                print(f'model dir path -- {middle_model_dir}')
                shutil.copytree(model_dir, middle_model_dir, dirs_exist_ok=True)
                print(f'New model {train_steps_current} has been saved !!!')

            if stop_training or (max_steps is not None and train_steps_current == max_steps):
                break

        if max_steps is not None and train_steps_current == max_steps:
            break

    # Update ema model at the end of training
    if ema_model is not None:
        # update ema
        if train_steps_current < step_start_ema:
            # reset parameters ema
            ema_model.load_state_dict(model.state_dict())
        ema.update_model_average(ema_model, model)

    # Save model at end of training
    save_nn_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                        epoch, train_steps_current, checkpoints_dir)
    save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)
    saved_main_folder = model_saving_address
    final_model_dir = os.path.join(saved_main_folder, 'final')
    os.makedirs(saved_main_folder, exist_ok=True)
    shutil.copytree(model_dir, final_model_dir)
    print(f'Final model has been saved !!!')

    print(f'\n------- TRAINING FINISHED -------')

