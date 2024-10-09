from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import casadi as ca
import control
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.models import ConditionedTemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn, ddpm_cart_pole_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params

import torch.nn as nn

import time
from multiprocessing import Pool, Manager, Array

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../nn_trained_models/' # '../../trained_models/' cart_pole_diffusion_based_on_MPD/nn_trained_models/nmpc_672000_training_data
MODEL_FOLDER = 'nmpc_672000_training_data_DIM_5' # choose a main model folder saved in the trained_models (eg. 420000 is the number of total training data, this folder contains all trained models based on the 420000 training data)
MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/nn_trained_models/nmpc_672000_training_data_DIM_5/final' # the absolute path of the trained model
MODEL_ID = 'final' # number of training

POSITION_INITIAL_RANGE = np.linspace(-0.5, 0.5,5) # np.linspace(-1,1,5)
THETA_INITIAL_RANGE = np.linspace(3*np.pi/4, 5*np.pi/4, 5) # np.linspace(-np.pi/4,np.pi/4,5)
WEIGHT_GUIDANC = 0.01 # non-conditioning weight
X0_IDX = 18 # range:[0,199] 20*20 data 0
ITERATIONS = 80 # control loop (steps) 50
HORIZON = 64 # mpc horizon 8

RESULTS_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/nn_nmpc_672000/final'

class AMPCNet_Inference(nn.Module):
    def __init__(self, input_size, output_size):
        super(AMPCNet_Inference, self).__init__()
        # Define the hidden layers and output layer
        self.hidden1 = nn.Linear(input_size, 2)  # First hidden layer with 2 neurons
        self.hidden2 = nn.Linear(2, 50)          # Second hidden layer with 50 neurons
        self.hidden3 = nn.Linear(50, 50)         # Third hidden layer with 50 neurons
        self.output = nn.Linear(50, output_size) # Output layer

    def forward(self, x, horizon):
        # print(x.dtype)  # Check the data type of the input tensor
        # print(self.hidden1.weight.dtype)  # Check the data type of the Linear layer's weights
        # Forward pass through the network with the specified activations
        x = x.to(torch.float32) 
        x = torch.tanh(self.hidden1(x))          # Tanh activation for first hidden layer
        x = torch.tanh(self.hidden2(x))          # Tanh activation for second hidden layer
        x = torch.tanh(self.hidden3(x))          # Tanh activation for third hidden layer
        x = self.output(x)                       # Linear activation (no activation function) for the output layer

        # reshape the output
        x = x.view(1, horizon, 1) # 1*horizon*1

        return x

# cart pole linear dynamics
def cart_pole_dynamics(x, u):

    A = np.array([
    [0, 1, 0, 0],
    [0, -0.1, 3, 0],
    [0, 0, 0, 1],
    [0, -0.5, 30, 0]
    ])

    B = np.array([
    [0],
    [2],
    [0],
    [5]
    ])

    C = np.eye(4)

    D = np.zeros((4,1))

    # state space equation
    sys_continuous = control.ss(A, B, C, D)

    # sampling time
    Ts = 0.1

    # convert to discrete time dynamics
    sys_discrete = control.c2d(sys_continuous, Ts, method='zoh')

    A_d = sys_discrete.A
    B_d = sys_discrete.B
    C_d = sys_discrete.C
    D_d = sys_discrete.D

    # States
    x_pos = x[0]
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    x_next = ca.vertcat(
        A_d[0,0]*x_pos + A_d[0,1]*x_dot + A_d[0,2]*theta + A_d[0,3]*theta_dot + B_d[0,0]*u,
        A_d[1,0]*x_pos + A_d[1,1]*x_dot + A_d[1,2]*theta + A_d[1,3]*theta_dot + B_d[1,0]*u,
        A_d[2,0]*x_pos + A_d[2,1]*x_dot + A_d[2,2]*theta + A_d[2,3]*theta_dot + B_d[2,0]*u,
        A_d[3,0]*x_pos + A_d[3,1]*x_dot + A_d[3,2]*theta + A_d[3,3]*theta_dot + B_d[3,0]*u,
    )
    return x_next



######### cart pole nonlinear dynamics ##########

# dynamic parameter
M_CART = 2.0
M_POLE = 1.0
M_TOTAL = M_CART + M_POLE
L_POLE = 1.0
MPLP = M_POLE*L_POLE
G = 9.81
MPG = M_POLE*G
MTG = M_TOTAL*G
MTLP = M_TOTAL*G
PI_2 = 2*np.pi
PI_UNDER_2 = 2/np.pi
PI_UNDER_1 = 1/np.pi

def EulerForwardCartpole_virtual_Casadi(dynamic_update_virtual_Casadi, dt, x,u) -> ca.vertcat:
    xdot = dynamic_update_virtual_Casadi(x,u)
    return x + xdot * dt

def EulerForwardCartpole_virtual_4DoF(dt, x,u) -> ca.vertcat:
    xdot = np.array([
        x[1],            # xdot 

        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        -PI_UNDER_2 * (x[2]-np.pi) * x[3],        # theta_star_dot

        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u
          )/(MTLP - MPLP*np.cos(x[2])**2) # thetaddot
        
        # -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    ])
    return x + xdot * dt

def dynamic_update_NMPC_cartpole_4DoF_Casadi(x, u) -> ca.vertcat:
    return ca.vertcat(
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        -PI_UNDER_2 * (x[2]-np.pi) * x[3],        # theta_star_dot
        
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u
          )/(MTLP - MPLP*np.cos(x[2])**2)  # thetaddot
        
        # -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    )

def dynamic_update_virtual_Casadi(x, u) -> ca.vertcat:
    # Return the derivative of the state
    # u is 1x1 array, covert to scalar by u[0] 
        
    return ca.vertcat(
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u[0] 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        x[3],        # thetadot
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u[0]
          )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    )

def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi
#################################################

################# casadi setting and NMPC solver #################
# Opt
opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

# NMPC
NUM_STATE = 5
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT]) #Q = np.diag([0.01, 0.01, Q_REDUNDANT, 0.01])
R = 0.001
P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT]) #P = np.diag([0.01, 0.1, P_REDUNDANT, 0.1])


TS = 0.01

# initial idx
IDX_X_INI = 0
IDX_THETA_INI = 1
IDX_THETA = 2
IDX_THETA_RED = 4

# initial guess
INITIAL_GUESS_NUM = 2
initial_guess_x = [5, 0]
initial_guess_u = [1000, -10000]

def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, ts: float, opts_setting ):
    # casadi_Opti
    optimizer_normal = ca.Opti()
    
    ##### normal mpc #####  
    # x and u mpc prediction along N
    X_pre = optimizer_normal.variable(num_state, horizon + 1) 
    U_pre = optimizer_normal.variable(1, horizon)
    # set intial guess
    optimizer_normal.set_initial(X_pre, initial_guess_x)
    optimizer_normal.set_initial(U_pre, initial_guess_u)

    optimizer_normal.subject_to(X_pre[:, 0] == x0)  # starting state

    # cost 
    cost = 0

    # initial cost
    cost += Q_cost[0,0]*X_pre[0, 0]**2 + Q_cost[1,1]*X_pre[1, 0]**2 + Q_cost[2,2]*X_pre[2, 0]**2 + Q_cost[3,3]*X_pre[3, 0]**2 + Q_cost[4,4]*X_pre[4, 0]**2

    # state cost
    for k in range(0,HORIZON-1):
        x_next = system_update(system_dynamic,ts,X_pre[:, k],U_pre[:, k])
        optimizer_normal.subject_to(X_pre[:, k + 1] == x_next)
        cost += Q_cost[0,0]*X_pre[0, k+1]**2 + Q_cost[1,1]*X_pre[1, k+1]**2 + Q_cost[2,2]*X_pre[2, k+1]**2 + Q_cost[3,3]*X_pre[3, k+1]**2 + Q_cost[4,4]*X_pre[4, k+1]**2 + R_cost * U_pre[:, k]**2

    # terminal cost
    x_terminal = system_update(system_dynamic,ts,X_pre[:, horizon-1],U_pre[:, horizon-1])
    optimizer_normal.subject_to(X_pre[:, horizon] == x_terminal)
    cost += P[0,0]*X_pre[0, HORIZON]**2 + P[1,1]*X_pre[1, HORIZON]**2 + P[2,2]*X_pre[2, HORIZON]**2 + P[3,3]*X_pre[3, HORIZON]**2 + P[4,4]*X_pre[4, HORIZON]**2 + R_cost * U_pre[:, HORIZON-1]**2

    optimizer_normal.minimize(cost)
    optimizer_normal.solver('ipopt',opts_setting)

    with TimerCUDA() as t_NMPC_sampling:
        sol = optimizer_normal.solve()
    print(f't_MPC_sampling: {t_NMPC_sampling.elapsed:.4f} sec')
    single_NMPC_time = np.round(t_NMPC_sampling.elapsed,4)
    # NMPC_total_time += single_MPC_time

    # sol = optimizer_normal.solve()
    X_sol = sol.value(X_pre)
    U_sol = sol.value(U_pre)
    Cost_sol = sol.value(cost)
    return X_sol, U_sol, Cost_sol, single_NMPC_time

# MPC_NormalData_Process


# EulerForwardCartpole_virtual
def EulerForwardCartpole_virtual(dt, x,u) -> ca.vertcat:
    xdot = np.array([
        x[1],            # xdot 
        ( MPLP * -np.sin(x[2]) * x[3]**2 
          +MPG * np.sin(x[2]) * np.cos(x[2])
          + u 
          )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

        x[3],        # thetadot
        ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
          -MTG * np.sin(x[2])
          -np.cos(x[2])*u
          )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    ])
    return x + xdot * dt

# MPC_NoiseData_Process


# RunMPCForSingle_IniState_IniGuess



@single_experiment_yaml
def experiment(
    #########################################################################################
    # Model id
    model_id: str = MODEL_FOLDER, 

    planner_alg: str = 'mpd',

    n_samples: int = 1,

    n_diffusion_steps_without_noise: int = 5,

    ##############################################################
    device: str = 'cuda',

    ##############################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = 'logs',
    ##############################################################
    # **kwargs
):
    ##############################################################

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ###############################################################
    print(f'##########################################################################################################')
    print(f'Model -- {model_id}')
    # print(f'Algorithm -- {planner_alg}')
    # run_prior_only = False
    # run_prior_then_guidance = False
    # if planner_alg == 'mpd':
    #     pass
    # elif planner_alg == 'diffusion_prior_then_guide':
    #     run_prior_then_guidance = True
    # elif planner_alg == 'diffusion_prior':
    #     run_prior_only = True
    # else:
    #     raise NotImplementedError

    ################################################################
    model_dir = MODEL_PATH 
    results_dir = os.path.join(model_dir, 'results_inference')
    
    os.makedirs(results_dir, exist_ok=True)

    # args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    #################################################################
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        dataset_subdir = 'CartPole-LMPC',
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    print(f'dataset -- {len(dataset)}')

    n_support_points = dataset.n_support_points
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    #################################################################
    # load initial starting state x0
    rng_x = POSITION_INITIAL_RANGE # 20 x_0 samples
    rng_theta = THETA_INITIAL_RANGE # 20 theta_0 samples
    
    # all possible initial states combinations
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
           rng0.append([m,n])
    rng0 = np.array(rng0,dtype=float)

    # one initial state for test
    # test = X0_IDX

    # x_0 = rng0[test,0]
    # x_0= round(x_0, 4)
    # theta_0 = rng0[test,1]
    # theta_0= round(theta_0, 4)

    x_0 = rng0[X0_IDX,IDX_X_INI]
    theta_0 = rng0[X0_IDX,IDX_THETA_INI]
    theta_red_0 = ThetaToRedTheta(theta_0)
    x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0]) # theta_red replace theta!!!


    #initial context
    # x0 = np.array([[x_0 , 0, theta_0, 0]])  # np.array([[x_0 , 0, theta_0, 0]])
    initial_state = x0  

    ############################################################################
    # sampling loop
    num_loop = ITERATIONS
    x_track = np.zeros((NUM_STATE, num_loop+1))
    u_track = np.zeros((1, num_loop))
    u_horizon_track = np.zeros((num_loop, HORIZON))
    x_horizon_track = np.zeros((num_loop, HORIZON+1, NUM_STATE))

    x_track[:,0] = x0
    x_updated_by_u = np.zeros((HORIZON+1, NUM_STATE))

    # time recording 
    NN_total_time = 0

    for i in range(0, num_loop):
        x0 = torch.tensor(x0).to(device) # load data to cuda

        hard_conds = None
        context = dataset.normalize_condition(x0)
        context_weight = WEIGHT_GUIDANC
        
        nn_input = context

        #########################################################################
        # Load prior model
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
        
        # load prior nn model
        input_size = NUM_STATE    # Define your input size based on your problem
        output_size = HORIZON    # Define your output size based on your problem (e.g., regression or single class prediction)
        model = AMPCNet_Inference(input_size, output_size)

        # load ema state dict
        model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth'),
            map_location=tensor_args['device'])
        )
        model = model.to(device)
        # # 'ema_model_current_state_dict.pth'
        # diffusion_model.load_state_dict(
        #     torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
        #     map_location=tensor_args['device'])
        # )
        # diffusion_model.eval()
        # model = diffusion_model

        # model = torch.compile(model)

        model.eval()


        ########
        # # Sample u with classifier-free-guidance (CFG) diffusion model
        # with TimerCUDA() as timer_model_sampling:
        #     inputs_normalized_iters = model.run_CFG(
        #         context, hard_conds, context_weight,
        #         n_samples=n_samples, horizon=n_support_points,
        #         return_chain=True,
        #         sample_fn=ddpm_cart_pole_sample_fn,
        #         n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
        #     )
        # print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
        # # t_total = timer_model_sampling.elapsed
        # ########
        # inputs_iters = dataset.unnormalize_states(inputs_normalized_iters)

        # inputs_final = inputs_iters[-1]

        with torch.no_grad():
            with TimerCUDA() as t_NN_sampling:
                nn_output = model(nn_input, horizon = HORIZON)
            print(f't_NN_sampling: {t_NN_sampling.elapsed:.4f} sec')
            single_NN_time = np.round(t_NN_sampling.elapsed,4)
            NN_total_time += single_NN_time
            inputs_final = dataset.unnormalize_states(nn_output)

        # print(f'control_inputs -- {inputs_final}')

        print(f'\n--------------------------------------\n')
        
        x0 = x0.cpu() # copy cuda tensor at first to cpu
        x0_array = np.squeeze(x0.numpy()) # matrix (1*4) to vector (4)
        horizon_inputs = np.zeros((1, HORIZON))
        inputs_final = inputs_final.cpu()
        for n in range(0,HORIZON):
            horizon_inputs[0,n] = round(inputs_final[0,n,0].item(),4)
        # print(f'horizon_inputs -- {horizon_inputs}')
        applied_input = round(inputs_final[0,0,0].item(),4) # retain 4 decimal places
        print(f'applied_input -- {applied_input}')

        # save the control input from diffusion sampling
        u_track[:,i] = applied_input
        u_horizon_track[i,:] = horizon_inputs

        # update states along the horizon
        x_updated_by_u[0,:] = x0
        x0_horizon = np.squeeze(x0.numpy())
        for z in range(0,horizon_inputs.shape[1]):
            #x_horizon_update = cart_pole_dynamics(x0_horizon, horizon_inputs[0,z])
            x_horizon_update = EulerForwardCartpole_virtual(TS,x0_horizon,horizon_inputs[0,z])
            x_updated_by_u[z+1,:] = x_horizon_update.T
            x0_horizon = x_horizon_update
        x_horizon_track[i,:,:] = np.round(x_updated_by_u, decimals=4)

        # update cart pole state
        x_next = EulerForwardCartpole_virtual(TS,x0_array,applied_input) # cart_pole_dynamics(x0_array, applied_input)
        print(f'x_next-- {x_next}')
        x0 = np.array(x_next)
        x0 = x0.T # transpose matrix

        # save the new state
        x_track[:,i+1] = x0

        # x_updated_by_u = np.zeros((HORIZON+1, 4))
        x_updated_by_u[0,:] = x0

        # save new starting state along the horizon
        # x_updated_by_u = np.zeros((HORIZON+1, 4))
        #x_updated_by_u[0,:] = x0



    # print all x and u 
    print(f'x_track-- {x_track.T}')
    print(f'u_track-- {u_track}')
 #-------------------------- Sampling finished --------------------------------



 ########################## MPC #################################

    # simulation time
    # T = 3.3  # Total time (seconds) 6.5
    # dt = 0.1  # Time step (seconds)
    # t = np.arange(0, T, dt) # time intervals 65
    # print(t.shape)

    N = HORIZON # prediction horizon
    
    x_nmpc_horizon_track = np.zeros((INITIAL_GUESS_NUM*num_loop, HORIZON+1, NUM_STATE)) # 2*80,64+1,4

    # # mpc parameters
    # Q = np.diag([10, 1, 10, 1]) 
    # R = np.array([[1]])
    # P = np.diag([100, 1, 100, 1])

    # Define the initial states range
    rng_x = POSITION_INITIAL_RANGE
    rng_theta = THETA_INITIAL_RANGE
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
            rng0.append([m,n])
    rng0 = np.array(rng0)
    num_datagroup = len(rng0)
    print(f'num_datagroup -- {num_datagroup}')

    # ##### data collecting loop #####

    # data set for each turn
    x_nmpc_track = np.zeros((NUM_STATE, INITIAL_GUESS_NUM*(num_loop+1))) # 4,2*(80+1)
    u_nmpc_track = np.zeros((INITIAL_GUESS_NUM, num_loop)) # 2,80
    u_nmpc_horizon_track = np.zeros((INITIAL_GUESS_NUM*num_loop, HORIZON)) # 2*80,64

    NMPC_total_time_with_NITIAL_GUESS_NUM = np.zeros((INITIAL_GUESS_NUM, 1))

    # NMPC initial setting
    # x_initial_guess = np.zeros((NUM_STATE, num_loop+1))

    # x_initial_guess_1 = initial_guess_x[0]
    # u_initial_guess_1 = initial_guess_u[0]

    # x_initial_guess_2 = initial_guess_x[1]
    # u_initial_guess_2 = initial_guess_u[1]

    # idx_group_of_initial_guess_1 = 0*num_datagroup+X0_IDX
    # idx_group_of_initial_guess_2 = 0*num_datagroup+X0_IDX

    x_0 = rng0[X0_IDX,IDX_X_INI]
    theta_0 = rng0[X0_IDX,IDX_THETA_INI]
    theta_red_0 = ThetaToRedTheta(theta_0)
    # x0 = np.array([x_0, 0.0, theta_red_0, 0])
    # print(f'x0-- {x0}')

    for idx_ini_guess in range(0, INITIAL_GUESS_NUM):
        x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
        print(f'x0-- {x0}')
        x_nmpc_track[:,idx_ini_guess*(num_loop+1)]  = x0
        NMPC_total_time = 0
        x_ini_guess = initial_guess_x[idx_ini_guess]
        u_ini_guess = initial_guess_u[idx_ini_guess]
        for i in range(0, num_loop):
             X_sol, U_sol, Cost_sol, single_MPC_time = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HORIZON, Q, R, TS, opts_setting)
             print(f'X_sol_shape -- {X_sol.shape}')
             print(f'U_sol - {U_sol}') 
            
             # applied u
             applied_u = U_sol[0]

             # x0 next 
             x0_next = EulerForwardCartpole_virtual(TS,x0,applied_u)
             x0 = x0_next
             print(f'x0_next-- {x0}')

             # new starting x and u
             #x_ini_guess = x0


             x_nmpc_track[:,idx_ini_guess*(num_loop+1)+(i+1)] = x0

             # x_nmpc_horizon_track[idx_ini_guess*num_loop+i,:,:] = np.round(X_sol.T, decimals=4)

             #save the first computed control input
             u_nmpc_track[idx_ini_guess,i] = U_sol[0]
            
             # save the computed control inputs along the mpc horizon
             u_nmpc_horizon_track[idx_ini_guess*num_loop+i,:] = U_sol
             
             # single solving time
             NMPC_total_time += single_MPC_time
        # solving time recording
        NMPC_total_time_with_NITIAL_GUESS_NUM[idx_ini_guess,0] = NMPC_total_time 
    
    # round u data
    u_nmpc_track = np.round(u_nmpc_track,decimals=4)
    u_nmpc_horizon_track = np.round(u_nmpc_horizon_track,decimals=4)
    print(f'u_nmpc_track -- {u_nmpc_track}')
    print(f'u_nmpc_horizon_track -- {u_nmpc_horizon_track}')


    # x_0 = rng0[test,0]
    # x_0= round(x_0, 4)
    # theta_0 = rng0[test,1]
    # theta_0= round(theta_0, 4)

    # #save the initial states
    # x0 = np.array([x_0, 0, theta_0, 0])  # Initial states
    # print(f'x0-- {x0}')
    # x_mpc_track[:,0] = x0

    ############# control loop ##################
    # for i in range(0, num_loop):
    #     # # casadi_Opti
    #     # optimizer = ca.Opti()

    #     # # x and u mpc prediction along N
    #     # X_pre = optimizer.variable(4, N + 1) 
    #     # print(X_pre)
    #     # U_pre = optimizer.variable(1, N) 

    #     # optimizer.subject_to(X_pre[:, 0] == x0)  # starting state

    #     # # cost 
    #     # cost = 0

    #     # # initial cost
    #     # cost += Q[0,0]*X_pre[0, 0]**2 + Q[1,1]*X_pre[1, 0]**2 + Q[2,2]*X_pre[2, 0]**2 + Q[3,3]*X_pre[3, 0]**2

    #     # # state cost
    #     # for k in range(0,N-1):
    #     #     x_next = cart_pole_dynamics(X_pre[:, k], U_pre[:, k])
    #     #     optimizer.subject_to(X_pre[:, k + 1] == x_next)
    #     #     cost += Q[0,0]*X_pre[0, k+1]**2 + Q[1,1]*X_pre[1, k+1]**2 + Q[2,2]*X_pre[2, k+1]**2 + Q[3,3]*X_pre[3, k+1]**2 + U_pre[:, k]**2

    #     # # terminal cost
    #     # x_terminal = cart_pole_dynamics(X_pre[:, N-1], U_pre[:, N-1])
    #     # optimizer.subject_to(X_pre[:, N] == x_terminal)
    #     # cost += P[0,0]*X_pre[0, N]**2 + P[1,1]*X_pre[1, N]**2 + P[2,2]*X_pre[2, N]**2 + P[3,3]*X_pre[3, N]**2 + U_pre[:, N-1]**2

    #     # optimizer.minimize(cost)
    #     # optimizer.solver('ipopt')
    #     # with TimerCUDA() as t_MPC_sampling:
    #     #     sol = optimizer.solve()
    #     # print(f't_MPC_sampling: {t_MPC_sampling.elapsed:.4f} sec')
    #     # single_MPC_time = np.round(t_MPC_sampling.elapsed,4)
    #     # MPC_total_time += single_MPC_time

    #     # X_sol = sol.value(X_pre)
    #     # # print(f'X_sol_shape -- {X_sol.shape}')
    #     # U_sol = sol.value(U_pre)
    #     # print(f'U_sol - {U_sol}')

    #     # initial guess
    #     for idx_ini_guess in range(0, INITIAL_GUESS_NUM): 
    #         for turn in range(0,num_datagroup):
    #             # initial guess
    #             x_ini_guess = initial_guess_x[idx_ini_guess]
    #             u_ini_guess = initial_guess_u[idx_ini_guess]
    #             # idx_group_of_control_step = idx_ini_guess*num_datagroup+turn

    #     X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HORIZON, Q, R, TS, opts_setting)

    #     print(f'X_sol -- {X_sol}')
    #     print(f'U_sol -- {U_sol}')
        
    #     # select the first updated states as new starting state ans save in the x_track
    #     x0 = X_sol[:,1]
    #     print(f'x0_new-- {x0}')
    #     x_mpc_track[:,i+1] = x0

    #     x_mpc_horizon_track[i,:,:] = np.round(X_sol.T, decimals=4)

    #     #save the first computed control input
    #     u_mpc_track[:,i] = U_sol[0]
        
    #     # save the computed control inputs along the mpc horizon
    #     u_mpc_horizon_track[i,:] = U_sol

    # ##### data collecting loop #####
    # data (x,u) collecting (saved in PT file)
    # SIZE_NORMAL_DATA = INITIAL_GUESS_NUM*num_datagroup*(ITERATIONS)
    # # SIZE_NOISE_DATA = INITIAL_GUESS_NUM*num_datagroup*ITERATIONS_X_NUMNOISY
    # x_normal_shape = (SIZE_NORMAL_DATA,NUM_STATE)
    # u_normal_shape = (SIZE_NORMAL_DATA,HORIZON,1)
    # j_normal_shape = (SIZE_NORMAL_DATA)
    # x_noise_shape = (SIZE_NOISE_DATA,NUM_STATE)
    # u_noise_shape = (SIZE_NOISE_DATA,HORIZON,1)
    # j_noise_shape = (SIZE_NOISE_DATA)

    # MAX_CORE_CPU = 30
    # start_time = time.time()
    # with Manager() as manager:
    #     x_normal_shared_memory = manager.list([[0.0] * x_normal_shape[1]] * x_normal_shape[0])
    #     u_normal_shared_memory = manager.list([[[0.0] for _ in range(u_normal_shape[1])] for _ in range(u_normal_shape[0])])
        # j_normal_shared_memory = manager.list([0.0] * j_normal_shape)
        # x_noise_shared_memory = manager.list([[0.0] * x_noise_shape[1]] * x_noise_shape[0])
        # u_noise_shared_memory = manager.list([[[0.0] for _ in range(u_noise_shape[1])] for _ in range(u_noise_shape[0])])
        # j_noise_shared_memory = manager.list([0.0] * j_noise_shape)
        
        # argument_each_group = []
        # for idx_ini_guess in range(0, INITIAL_GUESS_NUM): 
        #     for turn in range(0,num_datagroup):
        #         # initial guess
        #         x_ini_guess = initial_guess_x[idx_ini_guess]
        #         u_ini_guess = initial_guess_u[idx_ini_guess]
        #         idx_group_of_control_step = idx_ini_guess*num_datagroup+turn
                
        #         #initial states
        #         x_0 = rng0[turn,IDX_X_INI]
        #         theta_0 = rng0[turn,IDX_THETA_INI]
        #         theta_red_0 = ThetaToRedTheta(theta_0)
        #         x0 = np.array([x_0, 0.0, theta_red_0, 0]) # theta_red_0 replace theta_0!!!
                
        #         argument_each_group.append((x_ini_guess, u_ini_guess, idx_group_of_control_step, x0, 
        #                                     x_normal_shared_memory, u_normal_shared_memory)) # , j_normal_shared_memory,x_noise_shared_memory, u_noise_shared_memory, j_noise_shared_memory
   
        # with Pool(processes=MAX_CORE_CPU) as pool:
        #     pool.starmap(RunMPCForSingle_IniState_IniGuess, argument_each_group)
                
        # # shared memory with manager list
        # x_all_normal = torch.from_numpy(np.array(x_normal_shared_memory))
        # u_all_normal = torch.from_numpy(np.array(u_normal_shared_memory))
        # j_all_normal = torch.from_numpy(np.array(j_normal_shared_memory))
        
        # x_all_noisy = torch.from_numpy(np.array(x_noise_shared_memory))
        # u_all_noisy = torch.from_numpy(np.array(u_noise_shared_memory))
        # j_all_noisy = torch.from_numpy(np.array(j_noise_shared_memory))

        # # show the first saved u and x0
        # print(f'x_all_normal -- {x_all_normal.size()}')
        # print(f'u_all_normal -- {u_all_normal.size()}')

        # ##### data combing #####
        # # u combine u_normal + u_noisy
        # u_training_data = torch.cat((u_all_normal, u_all_noisy), dim=0)
        # print(f'u_training_data -- {u_training_data.size()}')

        # # x0 combine x_normal + x_noisy
        # x0_conditioning_data = torch.cat((x_all_normal, x_all_noisy), dim=0)
        # print(f'x0_conditioning_data -- {x0_conditioning_data.size()}')

        # # J combine j_normal + j_noisy
        # J_training_data = torch.cat((j_all_normal, j_all_noisy), dim=0)

        # data saving
        # torch.save(u_training_data, os.path.join(SAVE_PATH, U_DATA_NAME))
        # torch.save(x0_conditioning_data, os.path.join(SAVE_PATH, X0_CONDITION_DATA_NAME))
        # torch.save(J_training_data, os.path.join(SAVE_PATH, J_DATA_NAME))

    # end_time = time.time()

    # duration = end_time - start_time
    # print(f"Time taken for generating data: {duration} seconds")
        
    # u_mpc_track = np.round(u_mpc_track,decimals=4)
    # u_mpc_horizon_track = np.round(u_mpc_horizon_track,decimals=4)
    
    # print(f'u_mpc_track -- {u_mpc_track}')
    # print(f'u_mpc_horizon_track -- {u_mpc_horizon_track}')

    ########################## Diffusion & MPC Control Inputs Results Saving ################################

    results_folder = os.path.join(RESULTS_SAVED_PATH, 'model_'+ str(MODEL_ID), 'x0_'+ str(X0_IDX))
    os.makedirs(results_folder, exist_ok=True)
    
    # save the first u 
    diffusion_u = 'u_diffusion.npy'
    diffusion_u_path = os.path.join(results_folder, diffusion_u)
    np.save(diffusion_u_path, u_track)

    mpc_u = 'u_mpc.npy'
    mpc_u_path = os.path.join(results_folder, mpc_u)
    np.save(mpc_u_path, u_nmpc_track)

    # save the u along horizon
    diffusion_u_horizon = 'u_horizon_diffusion.npy'
    diffusion_u_horizon_path = os.path.join(results_folder, diffusion_u_horizon)
    np.save(diffusion_u_horizon_path, u_horizon_track)

    mpc_u_horizon = 'u_horizon_mpc.npy'
    mpc_u_horizon_path = os.path.join(results_folder, mpc_u_horizon)
    np.save(mpc_u_horizon_path, u_nmpc_horizon_track)

    ########################## Diffusion & MPC States Results Saving ################################
    # save diffusion states along horizon
    diffusion_states = 'x_diffusion_horizon.npy'
    diffusion_states_path = os.path.join(results_folder, diffusion_states)
    np.save(diffusion_states_path, x_horizon_track)

    # save mpc states along horizon
    # mpc_states = 'x_mpc_horizon.npy'
    # mpc_states_path = os.path.join(results_folder, mpc_states)
    # np.save(mpc_states_path, x_nmpc_horizon_track)

    ########################## plot ################################
    num_i = num_loop
    step = np.linspace(0,num_i,num_i+1)
    step_u = np.linspace(0,num_i-1,num_i)

    plt.figure(figsize=(10, 8))

    plt.subplot(5, 1, 1)
    plt.plot(step, x_track[0, :])
    plt.plot(step, x_nmpc_track[0, :])
    plt.legend(['Diffusion Sampling', 'MPC']) 
    plt.ylabel('Position (m)')
    plt.grid()

    plt.subplot(5, 1, 2)
    plt.plot(step, x_track[1, :])
    plt.plot(step, x_nmpc_track[1, :])
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(5, 1, 3)
    plt.plot(step, x_track[2, :])
    plt.plot(step, x_nmpc_track[2, :])
    plt.ylabel('Angle (rad)')
    plt.grid()

    plt.subplot(5, 1, 4)
    plt.plot(step, x_track[3, :])
    plt.plot(step, x_nmpc_track[3, :])
    plt.ylabel('Ag Velocity (rad/s)')
    plt.grid()

    plt.subplot(5, 1, 5)
    plt.plot(step_u, u_track.reshape(num_loop,))
    plt.plot(step_u, u_nmpc_track.reshape(num_loop,))
    plt.ylabel('Ctl Input (N)')
    plt.xlabel('Control Step')
    plt.grid()
    # plt.show()
    # save figure 
    figure_name = 'NMPC_NN_' + 'x0_' + str(X0_IDX) + 'steps_' + str(ITERATIONS) + '.png'
    figure_path = os.path.join(results_dir, figure_name)
    plt.savefig(figure_path)

    ######### Performance Check #########
    position_difference = np.sum(np.abs(x_track[0, :] - x_nmpc_track[0, :]))
    print(f'position_difference - {position_difference}')

    velocity_difference = np.sum(np.abs(x_track[1, :] - x_nmpc_track[1, :]))
    print(f'velocity_difference - {velocity_difference}')

    theta_difference = np.sum(np.abs(x_track[2, :] - x_nmpc_track[2, :]))
    print(f'theta_difference - {theta_difference}')

    thetaVel_difference = np.sum(np.abs(x_track[3, :] - x_nmpc_track[3, :]))
    print(f'thetaVel_difference - {thetaVel_difference}')

    u_difference = np.sum(np.abs(u_track.reshape(num_loop,) - u_nmpc_track.reshape(num_loop,)))
    print(f'u_difference - {u_difference}')

    print(f'initial_state -- {initial_state}')

    print(f'NN_total_time -- {NN_total_time}')
    print(f'NMPC_total_time_with_NITIAL_GUESS_NUM -- {NMPC_total_time_with_NITIAL_GUESS_NUM}')



if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)