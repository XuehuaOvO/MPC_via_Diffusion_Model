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
from mpd.trainer import get_dataset, get_model,get_specified_dataset
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params

import torch.nn as nn

import time
from multiprocessing import Pool, Manager, Array

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../nn_trained_models/' # '../../trained_models/' cart_pole_diffusion_based_on_MPD/nn_trained_models/nmpc_672000_training_data
MODEL_FOLDER = 'nn_120000' # choose a main model folder saved in the trained_models (eg. 420000 is the number of total training data, this folder contains all trained models based on the 420000 training data)
MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/nn_trained_models/nn_120000/final' # the absolute path of the trained model
MODEL_ID = 'final' # number of training

POSITION_INITIAL_RANGE = np.linspace(-0.5, 0.5,5) # np.linspace(-1,1,5) np.linspace(-0.5, 0.5,5) 
THETA_INITIAL_RANGE = np.linspace(1.8, 4.4, 5) # np.linspace(-np.pi/4,np.pi/4,5) np.linspace(3*np.pi/4, 5*np.pi/4, 5) np.linspace(1.8, 4.4, 5)
WEIGHT_GUIDANC = 0.01 # non-conditioning weight
X0_IDX = 24 # range:[0,199] 20*20 data 0
ITERATIONS = 50 # control loop (steps) 50
HORIZON = 64 # mpc horizon 8

RESULTS_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/nn_120000/final'
DATASET_SUBDIR = 'diff_mpc_2024' # 'diff_mpc_2024/5_6_noise_3'

# NN Model Class
class AMPCNet_Inference(nn.Module):
    def __init__(self, input_size, output_size):
        super(AMPCNet_Inference, self).__init__()
        # Define the hidden layers and output layer
        self.hidden1 = nn.Linear(input_size, 2)  # First hidden layer with 2 neurons
        self.hidden2 = nn.Linear(2, 50)          # Second hidden layer with 50 neurons
        self.hidden3 = nn.Linear(50, 50)         # Third hidden layer with 50 neurons
        self.output = nn.Linear(50, output_size) # Output layer

    def forward(self, x, horizon):
        # Forward pass through the network with the specified activations
        x = x.to(torch.float32) 
        x = torch.tanh(self.hidden1(x))          # Tanh activation for first hidden layer
        x = torch.tanh(self.hidden2(x))          # Tanh activation for second hidden layer
        x = torch.tanh(self.hidden3(x))          # Tanh activation for third hidden layer
        x = self.output(x)                       # Linear activation (no activation function) for the output layer

        # reshape the output
        x = x.view(1, horizon, 1) # 1*horizon*1

        return x


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


def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi


# calMPCCost
def calMPCCost(Q,R,P,u_hor:torch.tensor,x0:np.array, ModelUpdate_func, dt):
    # u 1x64x1, x0: ,state
    num_state = x0.shape[0]
    num_u = u_hor.size(0)
    num_hor = u_hor.size(1)
    cost = 0
    
    # initial cost
    for i in range(num_state):
        cost = cost + Q[i][i] * x0[i] ** 2

    
    for i in range(num_u):
        cost = cost + R * u_hor[i][0][0] ** 2
        
    x_cur = x0
    u_cur = u_hor[0][0][0]
    IdxLastU = num_hor-1
    # stage cost
    for i in range(1,IdxLastU):
        xnext = ModelUpdate_func(dt, x_cur, u_cur)
        unext = u_hor[:,i,0]
        for j in range(1,num_state):
            cost = cost + Q[j][j] * xnext[j] ** 2
        # for j in range(num_u):
        cost = cost + R * unext ** 2
        # update
        u_cur = unext
        x_cur = xnext
        
        
    #final cost
    for i in range(num_state):
        cost = cost + P[i][i] * xnext[i] ** 2
            
            
    return cost

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
initial_guess_x = [5, -5] # [5, 0]
initial_guess_u = [1000, -200] # [1000, -10000]



@single_experiment_yaml
def experiment(
    #########################################################################################
    # Model id
    model_id: str = MODEL_FOLDER, 

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

    ################################################################
    model_dir = MODEL_PATH 
    results_dir = os.path.join(model_dir, 'results_inference')
    
    os.makedirs(results_dir, exist_ok=True)

    #################################################################
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        dataset_subdir = DATASET_SUBDIR,
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

    x_0 = rng0[X0_IDX,IDX_X_INI]
    theta_0 = rng0[X0_IDX,IDX_THETA_INI] # rng0[X0_IDX,IDX_THETA_INI]
    theta_red_0 = ThetaToRedTheta(theta_0)
    x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0]) # x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])  np.array([0.5, 0.0, 3.1415926, 0, 3.1415926])
    print(f'x0  -- {x0}')


    #initial context
    # x0 = np.array([[x_0 , 0, theta_0, 0]])  # np.array([[x_0 , 0, theta_0, 0]])
    NN_initial_state = x0

    ############################### NN ###########################################
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

    # cost array
    cost_NN = np.zeros((1, ITERATIONS+1))
    cost_NMPC_pos = np.zeros((1, ITERATIONS+1))
    cost_NMPC_neg = np.zeros((1, ITERATIONS+1))

    # initial cost calculating
    cost_initial = Q[0,0]*x0[0]**2 + Q[1,1]*x0[1]**2 + Q[2,2]*x0[2]**2 + Q[3,3]*x0[3]**2 + Q[4,4]*x0[4]**2
    cost_NN[0,0] = cost_initial
    cost_NMPC_pos[0,0] = cost_initial
    cost_NMPC_neg[0,0] = cost_initial

    for i in range(0, num_loop):
        x0_NN_transform = np.copy(x0)
        x0_NN_transform[2] = x0_NN_transform[4]
        x0_NN = x0_NN_transform[:4] 

        print(f'x0_NN  -- {x0_NN}')
        x0_NN  = torch.tensor(x0_NN).to(device) # load data to cuda

        context = dataset.normalize_condition(x0_NN)
        
        nn_input = context

        #########################################################################

        # load prior nn model
        input_size = NUM_STATE-1    # Define your input size based on your problem
        output_size = HORIZON    # Define your output size based on your problem (e.g., regression or single class prediction)
        model = AMPCNet_Inference(input_size, output_size)

        # load ema state dict
        model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth'),
            map_location=tensor_args['device'])
        )
        model = model.to(device)

        model.eval()


        with torch.no_grad():
            with TimerCUDA() as t_NN_sampling:
                nn_output = model(nn_input, horizon = HORIZON)
            print(f't_NN_sampling: {t_NN_sampling.elapsed:.4f} sec')
            single_NN_time = np.round(t_NN_sampling.elapsed,4)
            NN_total_time += single_NN_time
            inputs_final = dataset.unnormalize_states(nn_output)

        # print(f'control_inputs -- {inputs_final}')

        print(f'\n--------------------------------------\n')
        
        # x0 = x0.cpu() # copy cuda tensor at first to cpu
        # x0_array = np.squeeze(x0.numpy()) # matrix (1*4) to vector (4)
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

        # cost of one step
        cost_NN[0,i+1] = calMPCCost(Q,R,P,inputs_final,x0, EulerForwardCartpole_virtual, TS)

        # update cart pole state
        x_next = EulerForwardCartpole_virtual(TS,x0,applied_input) # cart_pole_dynamics(x0_array, applied_input)
        print(f'x_next-- {x_next}')
        x0 = np.array(x_next)
        x0 = x0.T # transpose matrix

        # save the new state
        x_track[:,i+1] = x0

        # x_updated_by_u = np.zeros((HORIZON+1, 4))
        x_updated_by_u[0,:] = x0

        x0 = x0.T




    # print all x and u 
    print(f'x_track-- {x_track.T}')
    print(f'u_track-- {u_track}')
 #-------------------------- Sampling finished --------------------------------



 ########################## NMPC #################################


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


    x_0 = rng0[X0_IDX,IDX_X_INI]
    theta_0 = rng0[X0_IDX,IDX_THETA_INI] 
    theta_red_0 = ThetaToRedTheta(theta_0)


    for idx_ini_guess in range(0, INITIAL_GUESS_NUM):
        x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])  # x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])  np.array([0.5, 0.0, 3.1415926, 0, 3.1415926])
        print(f'x0-- {x0}')
        x_nmpc_track[:,idx_ini_guess*(num_loop+1)]  = x0
        NMPC_total_time = 0
        x_ini_guess = initial_guess_x[idx_ini_guess]
        u_ini_guess = initial_guess_u[idx_ini_guess]
        for i in range(0, num_loop):
             X_sol, U_sol, Cost_sol, single_MPC_time = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HORIZON, Q, R, TS, opts_setting)
             # print(f'X_sol_shape -- {X_sol.shape}')
             # print(f'U_sol - {U_sol}') 
            
             # applied u
             applied_u = U_sol[0]

             # cost per step
             if idx_ini_guess == 0:
                 #pos cost
                 u = torch.from_numpy(U_sol)
                 u = u.reshape(1,HORIZON,1)
                 cost_NMPC_pos[0,i+1] = calMPCCost(Q,R,P,u,x0, EulerForwardCartpole_virtual, TS)
             else:
                 # neg cost
                 u = torch.from_numpy(U_sol)
                 u = u.reshape(1,HORIZON,1)
                 cost_NMPC_neg[0,i+1] = calMPCCost(Q,R,P,u,x0, EulerForwardCartpole_virtual, TS)

             # x0 next 
             x0_next = EulerForwardCartpole_virtual(TS,x0,applied_u)
             x0 = x0_next
             print(f'x0_next-- {x0}')

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
    print(f'u_nmpc_track shape-- {u_nmpc_track.shape}')
    # print(f'u_nmpc_horizon_track -- {u_nmpc_horizon_track}')

    print(f'x_nmpc_track-- {x_nmpc_track.shape}')


    ########################## Diffusion & MPC Control Inputs Results Saving ################################

    results_folder = os.path.join(RESULTS_SAVED_PATH, 'model_'+ str(MODEL_ID), 'x0_'+ str(X0_IDX))
    # os.makedirs(results_folder, exist_ok=True)
    
    # # save the first u 
    # diffusion_u = 'u_diffusion.npy'
    # diffusion_u_path = os.path.join(results_folder, diffusion_u)
    # np.save(diffusion_u_path, u_track)

    # mpc_u = 'u_mpc.npy'
    # mpc_u_path = os.path.join(results_folder, mpc_u)
    # np.save(mpc_u_path, u_nmpc_track)

    # # save the u along horizon
    # diffusion_u_horizon = 'u_horizon_diffusion.npy'
    # diffusion_u_horizon_path = os.path.join(results_folder, diffusion_u_horizon)
    # np.save(diffusion_u_horizon_path, u_horizon_track)

    # mpc_u_horizon = 'u_horizon_mpc.npy'
    # mpc_u_horizon_path = os.path.join(results_folder, mpc_u_horizon)
    # np.save(mpc_u_horizon_path, u_nmpc_horizon_track)

    ########################## Diffusion & MPC States Results Saving ################################
    # save diffusion states along horizon
    diffusion_states = 'x_diffusion_horizon.npy'
    diffusion_states_path = os.path.join(results_folder, diffusion_states)
    np.save(diffusion_states_path, x_horizon_track)


    ########################## plot ################################
    num_i = num_loop
    step = np.linspace(0,num_i,num_i+1)
    step_u = np.linspace(0,num_i-1,num_i)

    plt.figure(figsize=(12,10))

    plt.subplot(7, 1, 1)
    plt.plot(step, x_track[0, :])
    plt.plot(step, x_nmpc_track[0, 0:ITERATIONS+1])
    plt.plot(step, x_nmpc_track[0, ITERATIONS+1:])
    plt.legend(['NN 4D Sampling', 'NMPC_pos', 'NMPC_neg']) 
    plt.ylabel('Position (m)')
    plt.grid()

    plt.subplot(7, 1, 2)
    plt.plot(step, x_track[1, :])
    plt.plot(step, x_nmpc_track[1, 0:ITERATIONS+1])
    plt.plot(step, x_nmpc_track[1, ITERATIONS+1:])
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(7, 1, 3)
    plt.plot(step, x_track[2, :])
    plt.plot(step, x_nmpc_track[2, 0:ITERATIONS+1])
    plt.plot(step, x_nmpc_track[2, ITERATIONS+1:])
    plt.ylabel('Theta (rad)')
    plt.grid()

    plt.subplot(7, 1, 4)
    plt.plot(step, x_track[3, :])
    plt.plot(step, x_nmpc_track[3, 0:ITERATIONS+1])
    plt.plot(step, x_nmpc_track[3, ITERATIONS+1:])
    plt.ylabel('Theta Dot (rad/s)')
    plt.grid()

    plt.subplot(7, 1, 5)
    plt.plot(step, x_track[4, :])
    plt.plot(step, x_nmpc_track[4, 0:ITERATIONS+1])
    plt.plot(step, x_nmpc_track[4, ITERATIONS+1:])
    plt.ylabel('Theta Star (rad/s)')
    plt.grid()

    plt.subplot(7, 1, 6)
    plt.plot(step_u, u_track.reshape(num_loop,)) 
    plt.plot(step_u, u_nmpc_track[0,:]) # u_nmpc_track.reshape(num_loop,)
    plt.plot(step_u, u_nmpc_track[1,:])
    plt.ylabel('Ctl Input (N)')
    # plt.xlabel('Control Step')
    plt.grid()

    plt.subplot(7, 1, 7)
    plt.plot(step, cost_NN.reshape(ITERATIONS+1,)) 
    plt.plot(step, cost_NMPC_pos.reshape(ITERATIONS+1,)) 
    plt.plot(step, cost_NMPC_neg.reshape(ITERATIONS+1,))
    plt.ylabel('Cost')
    plt.xlabel('Control Step')
    plt.grid()

    # plt.show()
    # save figure 
    figure_name = 'NMPC_NN_' + 'x0_' + str(X0_IDX) + 'steps_' + str(ITERATIONS) + '.png'
    figure_path = os.path.join(results_dir, figure_name)
    plt.savefig(figure_path)

    ######### Performance Check #########
    # position_difference = np.sum(np.abs(x_track[0, :] - x_nmpc_track[0, :]))
    # print(f'position_difference - {position_difference}')

    # velocity_difference = np.sum(np.abs(x_track[1, :] - x_nmpc_track[1, :]))
    # print(f'velocity_difference - {velocity_difference}')

    # theta_difference = np.sum(np.abs(x_track[2, :] - x_nmpc_track[2, :]))
    # print(f'theta_difference - {theta_difference}')

    # thetaVel_difference = np.sum(np.abs(x_track[3, :] - x_nmpc_track[3, :]))
    # print(f'thetaVel_difference - {thetaVel_difference}')

    # u_difference = np.sum(np.abs(u_track.reshape(num_loop,) - u_nmpc_track.reshape(num_loop,)))
    # print(f'u_difference - {u_difference}')

    print(f'initial_state -- {NN_initial_state}')

    print(f'NN_total_time -- {NN_total_time}')
    print(f'NMPC_total_time_with_NITIAL_GUESS_NUM -- {NMPC_total_time_with_NITIAL_GUESS_NUM}')



if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)