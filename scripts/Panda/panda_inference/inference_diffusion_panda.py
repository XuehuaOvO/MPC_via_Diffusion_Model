from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController
from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params

import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import random
import casadi as ca

# Trained Model Info
TRAINED_MODELS_DIR = '../../trained_models/' # main loader of all saved trained models
MODEL_FOLDER = 'panda_test4_113400 '  #'180000_training_data' # choose a folder in the trained_models (eg. 420000 is the number of total training data, this folder contains all trained models based on the 420000 training data)
MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/trained_models/panda_test4_113400/final' #'/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/trained_models/180000_training_data/100000' # the absolute path of the trained model
MODEL_ID = 'final' # number of training
RESULTS_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/180000set'

# Sampling data
NUM_SEED = 0
WEIGHT_GUIDANC = 0.01 # non-conditioning weight
TARGET_POS = np.array([0.3, 0.3, 0.5]).reshape(3, 1)

def main():
    num_seed = 0

    # load model path
    model_dir = MODEL_PATH 
    results_dir = os.path.join(model_dir, 'results_inference')
    os.makedirs(results_dir, exist_ok=True)
    
    # load device data
    device: str = 'cuda'
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # load trained model
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    # Sampling initial data
    ini_joint_states, ini_guess = sampling_data_generating()

    # panda mujoco
    panda = mujoco.MjModel.from_xml_path('/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/xml/mjx_scene.xml')
    data = mujoco.MjData(panda)
   
    # full initial state 20*1
    x0 = generating_ini_state(panda, data, ini_joint_states)

    # load trained diffusion model
    model = loading_model(dataset, args, tensor_args, model_dir)
    
    # conditioning info
    x0 = torch.tensor(x0).to(device) # load data to cuda
    hard_conds = None
    context = dataset.normalize_condition(x0)
    context_weight = WEIGHT_GUIDANC

    # sampling
    with TimerCUDA() as t_diffusion_time:
        inputs_normalized_iters = model.run_CFG(
            context, hard_conds, context_weight,
            n_samples=1, horizon=n_support_points,
            return_chain=True,
            sample_fn=ddpm_cart_pole_sample_fn,
            n_diffusion_steps_without_noise=5,
        )
    print(f't_model_sampling: {t_diffusion_time.elapsed:.4f} sec')
    print(inputs_normalized_iters.size())

    



################################################################################################################################################

# initial sampling data generating
def sampling_data_generating():
    np.random.seed(NUM_SEED)
    random.seed(NUM_SEED)

    # Gaussian noise for initial states generating
    mean_ini = 0
    std_dev_ini = 0.1
    
    # sampling joint states generating
    sampling_ini_states_list = []

    gaussian_noise_ini_joint_1 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    gaussian_noise_ini_joint_2 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    gaussian_noise_ini_joint_3 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    gaussian_noise_ini_joint_4 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    gaussian_noise_ini_joint_5 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    gaussian_noise_ini_joint_6 = np.round(np.random.normal(mean_ini, std_dev_ini),2)
    sampling_ini_states_list.append([gaussian_noise_ini_joint_1,gaussian_noise_ini_joint_2,gaussian_noise_ini_joint_3,gaussian_noise_ini_joint_4,gaussian_noise_ini_joint_5,gaussian_noise_ini_joint_6, 0])
    sampling_ini_states = np.array(sampling_ini_states_list) # 1*7

    # random initial u guess
    sampling_u_guess_list = []

    u_guess_4 = np.round(random.uniform(-2, 2),2)
    u_guess_5 = np.round(random.uniform(-2, 2),2)
    u_guess_7 = np.round(random.uniform(-2, 2),2)
    sampling_u_guess_list.append([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])
    sampling_ini_u_guess = np.array(sampling_u_guess_list) # 1*7

    return sampling_ini_states, sampling_ini_u_guess


# Jacobian Matrix
def compute_jacobian(model, data, tpoint):
    """Compute the Jacobian for the hand."""
    # Jacobian matrices for position (jacp) and orientation (jacr)
    jacp = np.zeros((3, model.nv))  # Position Jacobian
    jacr = np.zeros((3, model.nv))  # Orientation Jacobian
    
    body_id = 9

    # mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    mujoco.mj_jac(model, data, jacp, jacr, tpoint, body_id)
    
    return jacp, jacr


# initial state (20*1)
def generating_ini_state(panda, data, ini_joint_states):
    data.qpos[:7] = ini_joint_states
    mujoco.mj_step(panda, data)
    q_ini = np.array(data.qpos).reshape(-1, 1)
    q_dot_ini = np.array(data.qvel).reshape(-1, 1)
    x_ini = np.array(data.xpos[9,:]).reshape(-1, 1)
    
    # compute 'hand' x_dot 
    jacp, _ = compute_jacobian(panda, data, TARGET_POS)
    jacp = jacp[:, :7]
    x_dot_ini = ca.mtimes(jacp, q_dot_ini[:7])
   
    # full initial state 20*1
    x0 = np.zeros((20, 1))
    x0[:7] = q_ini[:7]
    x0[7:14] = q_dot_ini[:7]
    x0[14:17] = x_ini
    x0[17:20] = x_dot_ini

    x0 = x0.reshape(1,20)

    return x0


# load trained model
def loading_model(dataset, args, tensor_args, model_dir):
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=ConditionedTemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    # 'ema_model_current_state_dict.pth'
    diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
        map_location=tensor_args['device'])
    )
    diffusion_model.eval()
    model = diffusion_model

    model = torch.compile(model)
    
    return model
    



if __name__ == "__main__":
      main()