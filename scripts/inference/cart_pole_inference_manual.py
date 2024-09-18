from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import casadi as ca
import control
import numpy as np
import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment, run_experiment_dynamics
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerCartPole
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../data_trained_models/'


@single_experiment_yaml
def experiment(
    ########################################################################################################################
    # Experiment configuration
    # model_id: str = 'EnvDense2D-RobotPointMass',
    # model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass',
    # model_id: str = 'EnvSimple2D-RobotPointMass',
    # model_id: str = 'EnvSpheres3D-RobotPanda', model_id: str = 'EnvDense2D-RobotPointMass'
    model_id: str = 'CartPole_LMPC',

    # planner_alg: str = 'diffusion_prior',
    # planner_alg: str = 'diffusion_prior_then_guide',
    planner_alg: str = 'mpd',

    use_guide_on_extra_objects_only: bool = False,

    n_samples: int = 1,

    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 1,
    n_diffusion_steps_without_noise: int = 5,

    weight_grad_cost_collision: float = 1e-2, # 
    weight_grad_cost_smoothness: float = 1e-7,

    factor_num_interpolated_points_for_collision: float = 1.5, # 

    trajectory_duration: float = 5.0,  # currently fixed

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    render: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    # x0 = None,
    ########################################################################
    # **kwargs
):
    ########################################################################################################################
    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ########################################################################################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')
    run_prior_only = False
    run_prior_then_guidance = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_guide':
        run_prior_then_guidance = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
    else:
        raise NotImplementedError

    ########################################################################################################################
    # model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    model_dir = '/home/xiao/mpd-public/data_trained_models/CartPole-LMPC'
    results_dir = os.path.join(model_dir, 'results_inference', str(seed))
    
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    ########################################################################################################################
    # Load dataset with env, robot, task
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    # print(f'train -- {train_subset}')
    print(f'dataset -- {len(dataset)}')

    n_support_points = dataset.n_support_points
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    ########################################################################################################################
    # load model
    # diffusion_configs = dict(
    #     variance_schedule=args['variance_schedule'],
    #     n_diffusion_steps=args['n_diffusion_steps'],
    #     predict_epsilon=args['predict_epsilon'],
    # )
    # unet_configs = dict(
    #     state_dim=dataset.state_dim,
    #     n_support_points=dataset.n_support_points,
    #     unet_input_dim=args['unet_input_dim'],
    #     dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    # )
    # diffusion_model = get_model(
    #     model_class=args['diffusion_model_class'],
    #     model=ConditionedTemporalUnet(**unet_configs),
    #     tensor_args=tensor_args,
    #     **diffusion_configs,
    #     **unet_configs
    # )
    # # 'ema_model_current_state_dict.pth'
    # diffusion_model.load_state_dict(
    #     torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
    #     map_location=tensor_args['device'])
    # )
    # diffusion_model.eval()
    # model = diffusion_model
    # # print(f'Diffusion_Model -- {model}')

    # freeze_torch_model_params(model)
    # model = torch.compile(model)


    ########################################################################################################################
    # load initial context and context_mask
    rng_x = np.linspace(-1,1,50) # 50 x_0 samples
    rng_theta = np.linspace(-np.pi/4,np.pi/4,50) # 50 theta_0 samples
    
    # all possible initial states combinations
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
           rng0.append([m,n])
    rng0 = np.array(rng0,dtype=float)

    # one initial state for test
    test = 0

    x_0 = rng0[test,0]
    x_0= round(x_0, 3)
    theta_0 = rng0[test,1]
    theta_0= round(theta_0, 3)


    #initial context
    x0 = np.array([[3.79511, 4.13421, -0.536992, -0.565583]])  # np.array([[x_0 , 0, theta_0, 0]])  


    
    ##################################################################################################################
    # # sampling loop
    # num_loop = 15
    # x_track = np.zeros((4, num_loop+1))
    # u_track = np.zeros((1, num_loop))

    # x_track[:,0] = x0

    # for i in range(0, num_loop):
    x0 = torch.tensor(x0)
    # print(f'x0 -- {x0}')
    # print(f'x0 -- {x0.size()}')
    hard_conds = None
    context = dataset.normalize_condition(x0)
    context_mask  = torch.zeros(context.size(0),1).to(device) # context_mask=1: drop_prob in sampling = 1 and 0
    context_weight = 8

    ########################################################################################################################
    # Load prior model
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
    # print(f'Diffusion_Model -- {model}')

    freeze_torch_model_params(model)
    model = torch.compile(model)
    model.warmup_CFG(horizon=n_support_points, device=device, context = context, context_mask=context_mask)


    ########
    # Sample trajectories with the diffusion/cvae model
    with TimerCUDA() as timer_model_sampling:
        inputs_normalized_iters = model.run_CFG(
            context, hard_conds, context_weight,
            n_samples=n_samples, horizon=n_support_points,
            return_chain=True,
            sample_fn=ddpm_cart_pole_sample_fn,
            # **sample_fn_kwargs,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            # ddim=True
        )
    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed

    ########
    inputs_iters = dataset.unnormalize_states(inputs_normalized_iters)

    inputs_final = inputs_iters[-1]
    print(f'control_inputs -- {inputs_final}')

    # return inputs_final
    # trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(trajs_final, return_indices=True)

    print(f'\n--------------------------------------\n')
        
        # x0_array = np.squeeze(x0.numpy()) # matrix (1*4) to vector (4)
        # applied_input = round(inputs_final[0,0].item(),4) # retain 4 decimal places

        # # save the control input from diffusion sampling
        # u_track[:,i] = applied_input

        # # print(f'x0_array-- {x0_array}')
        # # print(f'applied_input-- {applied_input}')

        # x_next = cart_pole_dynamics(x0_array, applied_input)
        # print(f'x_next-- {x_next}')
        # x0 = np.array(x_next)
        # # x0 = round(x0,4)
        # x0 = x0.T # transpose matrix

        # # save the new state
        # x_track[:,i+1] = x0

    # print(f'x_track-- {x_track.T}')
    # print(f'u_track-- {u_track}')
    # # plot
    # num_i = num_loop
    # step = np.linspace(0,num_i+2,num_i+1)
    # step_u = np.linspace(0,num_i+1,num_i)

    # plt.figure(figsize=(10, 8))

    # plt.subplot(5, 1, 1)
    # plt.plot(step, x_track[0, :])
    # plt.ylabel('Position (m)')
    # plt.grid()

    # plt.subplot(5, 1, 2)
    # plt.plot(step, x_track[1, :])
    # plt.ylabel('Velocity (m/s)')
    # plt.grid()

    # plt.subplot(5, 1, 3)
    # plt.plot(step, x_track[2, :])
    # plt.ylabel('Angle (rad)')
    # plt.grid()

    # plt.subplot(5, 1, 4)
    # plt.plot(step, x_track[3, :])
    # plt.ylabel('Ag Velocity (rad/s)')
    # plt.grid()

    # plt.subplot(5, 1, 5)
    # plt.plot(step_u, u_track.reshape(num_loop,))
    # plt.ylabel('Ctl Input (N)')
    # plt.xlabel('Control Step')
    # plt.grid()
    # plt.show()




# def cart_pole_dynamics(x, u):
#     A = np.array([
#     [0, 1, 0, 0],
#     [0, -0.1, 3, 0],
#     [0, 0, 0, 1],
#     [0, -0.5, 30, 0]
#     ])

#     B = np.array([
#     [0],
#     [2],
#     [0],
#     [5]
#     ])

#     C = np.eye(4)

#     D = np.zeros((4,1))

#     # state space equation
#     sys_continuous = control.ss(A, B, C, D)

#     # sampling time
#     Ts = 0.1

#     # convert to discrete time dynamics
#     sys_discrete = control.c2d(sys_continuous, Ts, method='zoh')

#     A_d = sys_discrete.A
#     B_d = sys_discrete.B
#     C_d = sys_discrete.C
#     D_d = sys_discrete.D

#     # States
#     x_pos = x[0]
#     x_dot = x[1]
#     theta = x[2]
#     theta_dot = x[3]

#     x_next = ca.vertcat(
#         A_d[0,0]*x_pos + A_d[0,1]*x_dot + A_d[0,2]*theta + A_d[0,3]*theta_dot + B_d[0,0]*u,
#         A_d[1,0]*x_pos + A_d[1,1]*x_dot + A_d[1,2]*theta + A_d[1,3]*theta_dot + B_d[1,0]*u,
#         A_d[2,0]*x_pos + A_d[2,1]*x_dot + A_d[2,2]*theta + A_d[2,3]*theta_dot + B_d[2,0]*u,
#         A_d[3,0]*x_pos + A_d[3,1]*x_dot + A_d[3,2]*theta + A_d[3,3]*theta_dot + B_d[3,0]*u,
#     )
#     return x_next

if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)


# if __name__ == '__main__':
#     # Leave unchanged
#     x_current = np.array([[-1.0000,  0.0000, -0.7850,  0.0000]])

#     for i in range(0, 10):
      
#       seed = 30
#       model_dir = '/home/xiao/mpd-public/data_trained_models/CartPole-LMPC'
#       results_dir = os.path.join(model_dir, 'results_inference', str(seed))

#       inputs_final = run_experiment_dynamics(experiment,args={'seed': seed, 'results_dir': results_dir, 'x0': x_current})
#       print(f'inputs_final  -- {inputs_final}')
#       # inputs_final = inputs_final.cpu()
#       # inputs_final = inputs_final.detach().numpy()
#       inputs_final = inputs_final[0]
#       print(f'input_u -- {inputs_final}')

#       x_next = cart_pole_dynamics(x_current, inputs_final)
#       x_current = np.array([x_next])

# print(f'x_next -- {x_next}')
