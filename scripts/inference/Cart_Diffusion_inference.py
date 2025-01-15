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

allow_ops_in_compiled_graph()


TRAINED_MODELS_DIR = '../../trained_models/' # main loader of all saved trained models
MODEL_FOLDER = 'cart_pole_84000_test1'   # '180000_training_data' # choose a folder in the trained_models (eg. 420000 is the number of total training data, this folder contains all trained models based on the 420000 training data)
MODEL_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/trained_models/cart_pole_84000_test1/final' # the absolute path of the trained model
MODEL_ID = 'final' # number of training

POSITION_INITIAL_RANGE = np.linspace(-5,5,5) 
THETA_INITIAL_RANGE = np.linspace(3*np.pi/4,5*np.pi/4,5) 
WEIGHT_GUIDANC = 0.01 # non-conditioning weight
X0_IDX = 12 # range:[0,24] 5*5 fata 
ITERATIONS = 80 # control loop (steps)
HORIZON = HOR = 32 # mpc horizon

NUM_STATE = 5
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.001, Q_REDUNDANT])
R = 0.1
P = np.diag([0.01, 0.01, 0, 0.001, P_REDUNDANT])
IDX_X_INI = 0
IDX_THETA_INI = 1
IDX_THETA = 2
IDX_THETA_RED = 4
TS = 0.01

# initial guess
INITIAL_GUESS_NUM = 2
initial_guess_x = [10, -10]
initial_guess_u = [1000, -1000]

RESULTS_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/cart_pole_test1_84000'

# sampling time
SAMPLING_TIMES = 10

# lmpc cart pole dynamics
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


############### Dynamics Define ######################
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

M_car = 4.5
m_pole = 0.12
l_pendul = 0.14
k = 0.5
c = 0.002
I = (m_pole*l_pendul**2)/3
v_1 = (M_car + m_pole)/(I*(M_car + m_pole) + (l_pendul**2)*m_pole*M_car)
v_2 = (I + (l_pendul**2)*m_pole)/(I*(M_car + m_pole) + (l_pendul**2)*m_pole*M_car)


def EulerForwardCartpole_virtual_Casadi(dynamic_update_virtual_Casadi, dt, x,u) -> ca.vertcat:
    xdot = dynamic_update_virtual_Casadi(x,u)
    return x + xdot * dt

def dynamic_update_virtual_Casadi(x, u) -> ca.vertcat:
    # Return the derivative of the state
    # u is 1x1 array, covert to scalar by u[0] 
        
    # return ca.vertcat(
    #     x[1],            # xdot 
    #     ( MPLP * -np.sin(x[2]) * x[3]**2 
    #       +MPG * np.sin(x[2]) * np.cos(x[2])
    #       + u[0] 
    #       )/(M_TOTAL - M_POLE*np.cos(x[2]))**2, # xddot

    #     x[3],        # thetadot
    #     ( -MPLP * np.sin(x[2]) * np.cos(x[2]) * x[3]**2
    #       -MTG * np.sin(x[2])
    #       -np.cos(x[2])*u[0]
    #       )/(MTLP - MPLP*np.cos(x[2])**2),  # thetaddot
        
    #     -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    # )

    return ca.vertcat(
        x[1],            # xdot 

        -k*v_2*x[1] + ((l_pendul*m_pole)**2)*G*v_2*x[2]/(I + (l_pendul**2)*m_pole) - l_pendul*m_pole*c*v_2*x[3]/(I + (l_pendul**2)*m_pole) + v_2*u[0], #xddot

        x[3],        # thetadot

        -l_pendul*m_pole*k*v_1/(M_car+m_pole)*x[1] + l_pendul*m_pole*G*v_1*x[2] - c*v_1*x[3] + l_pendul*m_pole*v_1/(M_car+m_pole)*u[0], # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    )
 
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

    xdot_new = np.array([
        x[1],            # xdot 
        
        -k*v_2*x[1] + ((l_pendul*m_pole)**2)*G*v_2/(I + (l_pendul**2)*m_pole)*x[2] - l_pendul*m_pole*c*v_2/(I + (l_pendul**2)*m_pole)*x[3] + v_2*u, #xddot

        x[3],        # thetadot

        -l_pendul*m_pole*k*v_1/(M_car+m_pole)*x[1] + l_pendul*m_pole*G*v_1*x[2] - c*v_1*x[3] + l_pendul*m_pole*v_1/(M_car+m_pole)*u, # thetaddot
        
        -PI_UNDER_2 * (x[2]-np.pi) * x[3]   # theta_stat_dot
    ])

    return x + xdot_new* dt

def ThetaToRedTheta(theta):
    return (theta-np.pi)**2/-np.pi + np.pi

def MPC_Solve( system_update, system_dynamic, x0:np.array, initial_guess_x:float, initial_guess_u:float, num_state:int, horizon:int, Q_cost:np.array, R_cost:float, ts: float, opts_setting ):
    # casadi_Opti
    optimizer_normal = ca.Opti()
    
    ##### normal mpc #####
    # x and u mpc prediction along N
    X_pre = optimizer_normal.variable(num_state, horizon + 1) 
    U_pre = optimizer_normal.variable(1, horizon)
    # set intial guess
    optimizer_normal.set_initial(X_pre, initial_guess_x)
    # optimizer_normal.set_initial(X_pre[0], initial_guess_x)
    # optimizer_normal.set_initial(X_pre[1], 10*initial_guess_x)
    # optimizer_normal.set_initial(X_pre[2], initial_guess_x)
    # optimizer_normal.set_initial(X_pre[3], 10*initial_guess_x)
    # optimizer_normal.set_initial(X_pre[4], initial_guess_x)
    optimizer_normal.set_initial(U_pre, initial_guess_u)

    optimizer_normal.subject_to(X_pre[:, 0] == x0)  # starting state

    # cost 
    cost = 0

    # initial cost
    cost += Q_cost[0,0]*X_pre[0, 0]**2 + Q_cost[1,1]*X_pre[1, 0]**2 + Q_cost[2,2]*X_pre[2, 0]**2 + Q_cost[3,3]*X_pre[3, 0]**2 + Q_cost[4,4]*X_pre[4, 0]**2

    # state cost
    for k in range(0,HOR-1):
        x_next = system_update(system_dynamic,ts,X_pre[:, k],U_pre[:, k])
        optimizer_normal.subject_to(X_pre[:, k + 1] == x_next)
        cost += Q_cost[0,0]*X_pre[0, k+1]**2 + Q_cost[1,1]*X_pre[1, k+1]**2 + Q_cost[2,2]*X_pre[2, k+1]**2 + Q_cost[3,3]*X_pre[3, k+1]**2 + Q_cost[4,4]*X_pre[4, k+1]**2 + R_cost * U_pre[:, k]**2

    # terminal cost
    x_terminal = system_update(system_dynamic,ts,X_pre[:, horizon-1],U_pre[:, horizon-1])
    optimizer_normal.subject_to(X_pre[:, horizon] == x_terminal)
    cost += P[0,0]*X_pre[0, HOR]**2 + P[1,1]*X_pre[1, HOR]**2 + P[2,2]*X_pre[2, HOR]**2 + P[3,3]*X_pre[3, HOR]**2 + P[4,4]*X_pre[4, HOR]**2 + R_cost * U_pre[:, HOR-1]**2

    optimizer_normal.minimize(cost)
    optimizer_normal.solver('ipopt',opts_setting)
    sol = optimizer_normal.solve()
    X_sol = sol.value(X_pre)
    U_sol = sol.value(U_pre)
    Cost_sol = sol.value(cost)
    return X_sol, U_sol, Cost_sol

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

# Opt
opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

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

    ################################################################
    model_dir = MODEL_PATH 
    results_dir = os.path.join(model_dir, 'results_inference')
    
    os.makedirs(results_dir, exist_ok=True)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    #################################################################
    # Load dataset
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='InputsDataset',
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    print(f'dataset -- {len(dataset)}')

    n_support_points = dataset.n_support_points
    print(f'n_support_points -- {n_support_points}')
    print(f'state_dim -- {dataset.state_dim}')

    #################################################################
    # load initial starting state x0
     # for times in range(0,SAMPLING_TIMES): 
    rng_x = POSITION_INITIAL_RANGE # 20 x_0 samples
    rng_theta = THETA_INITIAL_RANGE # 20 theta_0 samples
    
    # all possible initial states combinations
    rng0 = []
    for m in rng_x:
        for n in rng_theta:
            rng0.append([m,n])
    rng0 = np.array(rng0,dtype=float)

    # one initial state for test
    test = X0_IDX

    x_0 = rng0[X0_IDX,IDX_X_INI]
    theta_0 = rng0[X0_IDX,IDX_THETA_INI] 
    theta_red_0 = ThetaToRedTheta(theta_0)
    x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
    print(f'x0  -- {x0}')

    #initial context
    # x0 = np.array([[x_0 , 0, theta_0, 0]])  # np.array([[x_0 , 0, theta_0, 0]]) 
    initial_state = x0   

    ############################################################################
    # sampling loop
    num_loop = ITERATIONS
    x_track = np.zeros((SAMPLING_TIMES, NUM_STATE, num_loop+1))
    u_track = np.zeros((SAMPLING_TIMES, 1, num_loop))
    u_horizon_track = np.zeros((num_loop, HORIZON))
    x_horizon_track = np.zeros((num_loop, HORIZON+1, NUM_STATE))

    for times in range(0, SAMPLING_TIMES): 
        x_track[times,:,0] = x0
    x_updated_by_u = np.zeros((HORIZON+1, NUM_STATE))

    # time recording 
    Diffusion_total_time = 0
    MPC_total_time = 0

    # cost array
    cost_D = np.zeros((SAMPLING_TIMES, 1, ITERATIONS+1))
    cost_NMPC_pos = np.zeros((1, ITERATIONS+1))
    cost_NMPC_neg = np.zeros((1, ITERATIONS+1))

    # initial cost calculating
    cost_initial = Q[0,0]*x0[0]**2 + Q[1,1]*x0[1]**2 + Q[2,2]*x0[2]**2 + Q[3,3]*x0[3]**2 + Q[4,4]*x0[4]**2
    cost_D[:,0,0] = cost_initial
    cost_NMPC_pos[0,0] = cost_initial
    cost_NMPC_neg[0,0] = cost_initial

    for times in range(SAMPLING_TIMES): 
        x0 = initial_state
        for i in range(0, num_loop):
            print(f'sampling, ctl -- {times, i}')
            x0_D= np.copy(x0)
            x0_D = torch.tensor(x0_D).to(device) # load data to cuda

            hard_conds = None
            context = dataset.normalize_condition(x0_D.reshape(1,NUM_STATE))
            context_weight = WEIGHT_GUIDANC

            #########################################################################
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

            model = torch.compile(model)


            ########
            # Sample u with classifier-free-guidance (CFG) diffusion model
            with TimerCUDA() as t_diffusion_time:
                inputs_normalized_iters = model.run_CFG(
                    context, hard_conds, context_weight,
                    n_samples=n_samples, horizon=n_support_points,
                    return_chain=True,
                    sample_fn=ddpm_cart_pole_sample_fn,
                    n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
                )
            print(f't_model_sampling: {t_diffusion_time.elapsed:.4f} sec')
            single_Diffusion_time = np.round(t_diffusion_time.elapsed,4)
            Diffusion_total_time += single_Diffusion_time
            # t_total = timer_model_sampling.elapsed

            ########
            inputs_iters = dataset.unnormalize_states(inputs_normalized_iters)

            inputs_final = inputs_iters[-1]
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
            u_track[times,:,i] = applied_input
            # u_horizon_track[i,:] = horizon_inputs

            # cost of one step
            cost_D[times,0,i+1] = calMPCCost(Q,R,P,inputs_final,x0, EulerForwardCartpole_virtual, TS)

            # update states along the horizon
            # x_updated_by_u[0,:] = x0
            # x0_horizon = np.squeeze(x0.numpy())
            # for z in range(0,horizon_inputs.shape[1]):
            #     x_horizon_update = cart_pole_dynamics(x0_horizon, horizon_inputs[0,z])
            #     x_updated_by_u[z+1,:] = x_horizon_update.T
            #     x0_horizon = x_horizon_update
            # x_horizon_track[i,:,:] = np.round(x_updated_by_u, decimals=4)

            # update cart pole state
            x_next = EulerForwardCartpole_virtual(TS,x0,applied_input)
            print(f'x_next-- {x_next}')
            x0 = np.array(x_next)
            x0 = x0.T # transpose matrix

            # save the new state
            x_track[times,:,i+1] = x0

            #  x_updated_by_u = np.zeros((HORIZON+1, 4))
            x_updated_by_u[0,:] = x0
            
            x0 = x0.T
            # save new starting state along the horizon
            # x_updated_by_u = np.zeros((HORIZON+1, 4))
            # x_updated_by_u[0,:] = x0



    # print all x and u 
    # print(f'x_track-- {x_track.size()}')
    # print(f'u_track-- {u_track.size()}')
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
             X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HORIZON, Q, R, TS, opts_setting)
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
             # NMPC_total_time += single_MPC_time
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
    # np.save(mpc_u_path, u_mpc_track)

    # # save the u along horizon
    # diffusion_u_horizon = 'u_horizon_diffusion.npy'
    # diffusion_u_horizon_path = os.path.join(results_folder, diffusion_u_horizon)
    # np.save(diffusion_u_horizon_path, u_horizon_track)

    # mpc_u_horizon = 'u_horizon_mpc.npy'
    # mpc_u_horizon_path = os.path.join(results_folder, mpc_u_horizon)
    # np.save(mpc_u_horizon_path, u_mpc_horizon_track)

    ########################## Diffusion & MPC States Results Saving ################################
    # save diffusion states along horizon
    # diffusion_states = 'x_diffusion_horizon.npy'
    # diffusion_states_path = os.path.join(results_folder, diffusion_states)
    # np.save(diffusion_states_path, x_horizon_track)

    # # save mpc states along horizon
    # mpc_states = 'x_mpc_horizon.npy'
    # mpc_states_path = os.path.join(results_folder, mpc_states)
    # np.save(mpc_states_path, x_mpc_horizon_track)

    ########################## plot ################################
    num_i = num_loop
    step = np.linspace(0,num_i,num_i+1)
    step_u = np.linspace(0,num_i-1,num_i)

    plt.figure(figsize=(10,14))

    plt.subplot(6, 1, 1)
    plt.plot(step, x_nmpc_track[0, 0:ITERATIONS+1],label=f'NMPC (pos guess)',linewidth=7, color = 'gold')
    plt.plot(step, x_nmpc_track[0, ITERATIONS+1:],label=f'NMPC (neg guess)', linewidth=7, color = 'lightpink')
    for i in range(0, SAMPLING_TIMES):
        plt.plot(step, x_track[i, 0, :], color='deepskyblue')
    plt.plot(step, x_track[0, 0, :], label=f"Diffusion", color='deepskyblue')
    plt.legend() 
    # plt.legend(['Diffusion Sampling', 'NMPC_pos', 'NMPC_neg']) 
    plt.ylabel('Position (m)')
    plt.grid()

    plt.subplot(6, 1, 2)
    plt.plot(step, x_nmpc_track[1, 0:ITERATIONS+1],linewidth=7, color = 'gold')
    plt.plot(step, x_nmpc_track[1, ITERATIONS+1:],linewidth=7, color = 'lightpink')
    for i in range(0, SAMPLING_TIMES):
        plt.plot(step, x_track[i, 1, :], color='deepskyblue')
    # plt.plot(step, x_track[1, :])
    plt.ylabel('Velocity (m/s)')
    plt.grid()

    plt.subplot(6, 1, 3)
    plt.plot(step, x_nmpc_track[2, 0:ITERATIONS+1],linewidth=7, color = 'gold')
    plt.plot(step, x_nmpc_track[2, ITERATIONS+1:],linewidth=7, color = 'lightpink')
    for i in range(0,SAMPLING_TIMES):
        plt.plot(step, x_track[i, 2, :], color='deepskyblue')
    # plt.plot(step, x_track[2, :])
    plt.ylabel('Theta (rad)')
    plt.grid()

    plt.subplot(6, 1, 4)
    plt.plot(step, x_nmpc_track[3, 0:ITERATIONS+1],linewidth=7, color = 'gold')
    plt.plot(step, x_nmpc_track[3, ITERATIONS+1:],linewidth=7, color = 'lightpink')
    for i in range(0, SAMPLING_TIMES):
        plt.plot(step, x_track[i, 3, :], color='deepskyblue')
    # plt.plot(step, x_track[3, :])
    plt.ylabel('Theta Dot (rad/s)')
    plt.grid()

    plt.subplot(6, 1, 5)
    plt.plot(step, x_nmpc_track[4, 0:ITERATIONS+1],linewidth=7, color = 'gold')
    plt.plot(step, x_nmpc_track[4, ITERATIONS+1:],linewidth=7, color = 'lightpink')
    for i in range(0, SAMPLING_TIMES):
        plt.plot(step, x_track[i, 4, :], color='deepskyblue')
    # plt.plot(step, x_track[4, :])
    plt.ylabel('Theta Star (rad/s)')
    plt.grid()

    plt.subplot(6, 1, 6)
    plt.plot(step_u, u_nmpc_track[0,:],linewidth=7, color = 'gold') # u_nmpc_track.reshape(num_loop,)
    plt.plot(step_u, u_nmpc_track[1,:],linewidth=7, color = 'lightpink')
    for i in range(0,SAMPLING_TIMES):
        plt.plot(step_u, u_track[i, 0, :], color='deepskyblue')
    # plt.plot(step_u, u_track.reshape(num_loop,)) 
    plt.ylabel('Ctl Input (N)')
    plt.xlabel('Control Step')
    plt.grid()

    # plt.subplot(7, 1, 7)
    # plt.plot(step, cost_D.reshape(ITERATIONS+1,)) 
    # plt.plot(step, cost_NMPC_pos.reshape(ITERATIONS+1,)) 
    # plt.plot(step, cost_NMPC_neg.reshape(ITERATIONS+1,))
    # plt.ylabel('Cost')
    # plt.xlabel('Control Step')
    # plt.grid()

    # plt.show()
    # save figure 
    figure_name = 'Diffusion_CartPole_' + 'x0_' + str(X0_IDX) + 'steps_' + str(ITERATIONS) + '_diffusion_update_plot_' + '.pdf'
    figure_path = os.path.join(results_dir, figure_name)
    plt.savefig(figure_path)
    plt.show()

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

    print(f'initial_state -- {initial_state}')

    print(f'NN_total_time -- {Diffusion_total_time}')
    print(f'NMPC_total_time_with_NITIAL_GUESS_NUM -- {NMPC_total_time_with_NITIAL_GUESS_NUM}')



if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)