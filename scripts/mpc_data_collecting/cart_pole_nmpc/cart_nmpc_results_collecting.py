import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Manager, Array
import multiprocessing

############### Seetings ######################
# Attention: this py file can only set the initial range of position and theta, initial x_dot and theta_dot are always 0

MAX_CORE_CPU = 25

# data saving folder
SAVE_PATH =  "/MPC_DynamicSys/sharedVol/train_data/nmpc/multi_normal"
FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data_collecting/nmpc_cart_pole_collecting/cartpole_plots' # '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data_collecting/nmpc_cart_pole_collecting'

# control steps
CONTROL_STEPS = 80

# data range
NUM_INITIAL_X = 2
POSITION_INITIAL_RANGE = np.linspace(-5,5,NUM_INITIAL_X) 

NUM_INIYIAL_THETA = 2
THETA_INITIAL_RANGE = np.linspace(3*np.pi/4,5*np.pi/4,NUM_INIYIAL_THETA) 

# number of noisy data for each state
NUM_NOISY_DATA =20
NOISE_MEAN = 0
NOISE_SD = 0.15
CONTROLSTEP_X_NUMNOISY = CONTROL_STEPS*NUM_NOISY_DATA

HOR = 32 # mpc prediction horizon


# initial guess
INITIAL_GUESS_NUM = 2
initial_guess_x = [10, -10]
initial_guess_u = [1000, -1000]
# initial_guess_x = [5, 0]
# initial_guess_u = [1000, -10000]

# save data round to 4 digit
ROUND_DIG = 6

IDX_X_INI = 0
IDX_THETA_INI = 1
IDX_THETA = 2
IDX_THETA_RED = 4

# trainind data files name
filename_idx = '_ini_'+str(NUM_INITIAL_X)+'x'+str(NUM_INIYIAL_THETA)+'_noise_'+str(NUM_NOISY_DATA)+'_step_'+str(CONTROL_STEPS)+'_hor_'+str(HOR)+'.pt'
U_DATA_NAME = 'u' + filename_idx # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'x0' + filename_idx # 400000-4: tensor size for conditioning data in training
J_DATA_NAME = 'j'+ filename_idx

np.random.seed(42)

############# MPC #####################

# mpc parameters
NUM_STATE = 5
Q_REDUNDANT = 1000.0
P_REDUNDANT = 1000.0
Q = np.diag([0.01, 0.01, 0, 0.001, Q_REDUNDANT])
R = 0.1
P = np.diag([0.01, 0.01, 0, 0.001, P_REDUNDANT])
# Q = np.diag([0.01, 0.01, 0, 0.01, Q_REDUNDANT])
# R = 0.001
# P = np.diag([0.01, 0.1, 0, 0.1, P_REDUNDANT])


TS = 0.01

# Define the initial states range
rng_x = POSITION_INITIAL_RANGE 
rng_theta = THETA_INITIAL_RANGE 
rng0 = []
for idx in rng_x:
    for n in rng_theta:
        rng0.append([idx,n])
rng0 = np.array(rng0)
num_datagroup = len(rng0)
print(f'rng0 -- {rng0.shape}')


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


# Opt
opts_setting = {'ipopt.max_iter':20000, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}


#################################### single process ############################################

def MPC_NormalData_Process(x0, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_result_normal, j_result_normal, x_result_normal, idx_control_step=0) -> float:
    u_for_normal_x = np.zeros(HOR)
    
    X_sol, U_sol, Cost_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, x0, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, TS, opts_setting)
    u_for_normal_x = U_sol.reshape(HOR,1)
    j_for_normal_x = np.array(Cost_sol)
    # save normal x,u,j data in 0th step 
    idx_0step_normal_data = idx_control_step
    
    
    # shared memory with manager list
    x_result_normal[idx_0step_normal_data,:] = x0
    u_result_normal[idx_0step_normal_data,:,0] = u_for_normal_x.reshape(HOR)
    j_result_normal[idx_0step_normal_data,0] = j_for_normal_x
    
    # print normal
    print('-----------------------------------------normal result--------------------------------------------------------')
    print(f'(idx_ini_guess*num_datagroup+turn, control step) -- {idx_group_of_control_step, idx_control_step}')
    # print(f'u_sol-- {U_sol[0]}')
    # print(f'x0_new-- {X_sol[:,0]}')
    # print(f'cost-- {Cost_sol}')
    
    return U_sol[0]


def MPC_NoiseData_Process(x0, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_random_memory, x_random_memory, j_random_memory, idx_control_step=0, bAll = False):
    # noisey_x = np.zeros((NUM_NOISY_DATA, NUM_STATE))
    # u_for_noisy_x = np.zeros((NUM_NOISY_DATA, HOR, 1))
    # j_for_noisy_x = np.zeros((NUM_NOISY_DATA))
    for idx_noisy in range(0,NUM_NOISY_DATA):
        if (bAll == True): 
            noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = NUM_STATE)
            noisy_state = x0 + noise
        else:
            noise = np.random.normal(NOISE_MEAN, NOISE_SD, size = (1,2))
            noisy_state = x0 + [noise[0,0], 0, noise[0,1],0,0]
        
        noisy_state[IDX_THETA_RED] = ThetaToRedTheta(noisy_state[IDX_THETA])
        x_random_memory[idx_control_step*NUM_NOISY_DATA+idx_noisy,:] = noisy_state
        X_noise_sol, U_noisy_sol, Cost_noise_sol = MPC_Solve(EulerForwardCartpole_virtual_Casadi, dynamic_update_virtual_Casadi, noisy_state, x_ini_guess, u_ini_guess, NUM_STATE, HOR, Q, R, TS, opts_setting)
        
        # gey u, j by x
        u_random_memory[idx_control_step*NUM_NOISY_DATA+idx_noisy,:,0] = U_noisy_sol
        j_random_memory[idx_control_step*NUM_NOISY_DATA+idx_noisy,0] = Cost_noise_sol


    # save noise x,u,j data in 0th step 
    # idx_start_0step_nosie_data = idx_group_of_control_step*CONTROLSTEP_X_NUMNOISY + idx_control_step*NUM_NOISY_DATA
    # idx_end_0step_nosie_data = idx_start_0step_nosie_data + NUM_NOISY_DATA
    
    # # shared memory with manager list
    # x_result_noise[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = noisey_x.tolist()
    # u_result_noise[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = u_for_noisy_x.tolist()
    # j_result_noise[idx_start_0step_nosie_data:idx_end_0step_nosie_data] = j_for_noisy_x.tolist()


def RunMPCForSingle_IniState_IniGuess(x_ini_guess: float, u_ini_guess:float,idx_group_of_control_step:int,x0_state:np.array, 
                                      u_ini_memory: np.array, x_ini_memory: np.array, j_ini_memory: np.array,
                                      u_random_memory: np.array, x_random_memory: np.array, j_random_memory: np.array):

    ################ generate data for 0th step ##########################################################
    try:
        # normal at x0
        step = 0
        u0 = MPC_NormalData_Process(x0_state, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_ini_memory, j_ini_memory, x_ini_memory)
        
        # noisy at x0
        MPC_NoiseData_Process(x0_state, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_random_memory, x_random_memory, j_random_memory)
        
        ############################################## generate data for control step loop ##############################################
        # main mpc loop
        for idx_control_step in range(1, CONTROL_STEPS):
            #system dynamic update x 
            x0_next = EulerForwardCartpole_virtual(TS,x0_state,u0)
            
            ################################################# normal mpc loop to update state #################################################
            u0_cur = MPC_NormalData_Process(x0_next, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_ini_memory, j_ini_memory, x_ini_memory, idx_control_step)

            ################################## noise  ##################################
            MPC_NoiseData_Process(x0_next, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_random_memory, x_random_memory, j_random_memory, idx_control_step, True)
            
            # update
            x0_state = x0_next
            u0 = u0_cur
        

        #################### data saving ####################
        # to tensor
        torch_u_ini_memory_tensor = torch.Tensor(u_ini_memory)
        torch_u_random_memory_tensor = torch.Tensor(u_random_memory)
        torch_x_ini_memory_tensor = torch.Tensor(x_ini_memory)
        torch_x_random_memory_tensor = torch.Tensor(x_random_memory)
        torch_j_ini_memory_tensor = torch.Tensor(j_ini_memory)
        torch_j_random_memory_tensor = torch.Tensor(j_random_memory)

        # torch.save(torch_u_ini_memory_tensor, os.path.join(FOLDER_PATH , f'pure_u_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(torch_x_ini_memory_tensor , os.path.join(FOLDER_PATH , f'pure_x_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(torch_j_ini_memory_tensor, os.path.join(FOLDER_PATH , f'pure_j_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        
        # cat
        u_data = torch.cat((torch_u_ini_memory_tensor, torch_u_random_memory_tensor), dim=0)
        x_data = torch.cat((torch_x_ini_memory_tensor, torch_x_random_memory_tensor), dim=0)
        j_data = torch.cat((torch_j_ini_memory_tensor, torch_j_random_memory_tensor), dim=0)

        print(f'u_size -- {u_data.size()}')
        print(f'x_size -- {x_data.size()}')
        print(f'j_size -- {j_data.size()}')

        # save data in PT file for training
        # torch.save(u_data, os.path.join(FOLDER_PATH , f'u_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(x_data, os.path.join(FOLDER_PATH , f'x_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(j_data, os.path.join(FOLDER_PATH , f'j_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))

        # plots
        t = np.arange(0, CONTROL_STEPS*TS, TS) # np.arange(len(joint_states[1])) * panda.opt.timestep
        print(f't -- {len(t)}')

        # plot 1: 5 states
        plt.figure()
        # for i in range(5):
        plt.plot(t, x_ini_memory[:,0], label=f"x")
        plt.plot(t, x_ini_memory[:,1], label=f"x_dot")
        plt.plot(t, x_ini_memory[:,2], label=f"theta")
        plt.plot(t, x_ini_memory[:,3], label=f"theta_dot")
        plt.plot(t, x_ini_memory[:,4], label=f"theta_star")
        for z in range(CONTROL_STEPS):
                for i in range(5):
                    noisy_state_each_ctl_step = x_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,i]
                    for k in range(0,NUM_NOISY_DATA):
                            plt.scatter(t[z], noisy_state_each_ctl_step[k], s = 20, color = 'lightgrey')
        plt.scatter(t[CONTROL_STEPS-1], noisy_state_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        plt.xlabel("Time [s]")
        plt.ylabel("state")
        plt.legend()
        figure_name = 'idx-' + str(idx_group_of_control_step) + '_x' + '.pdf'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # plot 2: u
        plt.figure()
        plt.plot(t, u_ini_memory[:,0,0], label=f"u")
        for z in range(CONTROL_STEPS):
                noisy_u_each_ctl_step = u_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,0,0]
                for k in range(0, NUM_NOISY_DATA):
                    plt.scatter(t[z], noisy_u_each_ctl_step[k], s = 20, color = 'lightgrey')
        plt.scatter(t[CONTROL_STEPS-1], noisy_u_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        plt.xlabel("Time [s]")
        plt.ylabel("control input")
        plt.legend()
        figure_name = 'idx-' + str(idx_group_of_control_step) + '_u' + '.pdf'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

        # plot 3: cost
        plt.figure()
        plt.plot(t, j_ini_memory[:,0])
        for z in range(CONTROL_STEPS):
                noisy_j_each_ctl_step = j_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,0]
                for k in range(0, NUM_NOISY_DATA):
                    plt.scatter(t[z], noisy_j_each_ctl_step[k], s = 20, color = 'lightgrey')
        plt.scatter(t[CONTROL_STEPS-1], noisy_j_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        plt.xlabel("Time [s]")
        plt.ylabel("cost")
        plt.legend()
        figure_name = 'idx-' + str(idx_group_of_control_step) + '_j' + '.pdf'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)

            
            

    
    except Exception as e:
        print(f"Error: {e}")


######################################################################################################################
# ##### data collecting loop #####
# data (x,u) collecting (saved in PT file)
SIZE_NORMAL_DATA = INITIAL_GUESS_NUM*num_datagroup*(CONTROL_STEPS)
SIZE_NOISE_DATA = INITIAL_GUESS_NUM*num_datagroup*CONTROLSTEP_X_NUMNOISY
x_normal_shape = (SIZE_NORMAL_DATA,NUM_STATE)
u_normal_shape = (SIZE_NORMAL_DATA,HOR,1)
j_normal_shape = (SIZE_NORMAL_DATA)
x_noise_shape = (SIZE_NOISE_DATA,NUM_STATE)
u_noise_shape = (SIZE_NOISE_DATA,HOR,1)
j_noise_shape = (SIZE_NOISE_DATA)

def sub_main():
    # start_time = time.time()
    with Manager() as manager:
        x_normal_shared_memory = manager.list([[0.0] * x_normal_shape[1]] * x_normal_shape[0])
        u_normal_shared_memory = manager.list([[[0.0] for _ in range(u_normal_shape[1])] for _ in range(u_normal_shape[0])])
        j_normal_shared_memory = manager.list([0.0] * j_normal_shape)
        x_noise_shared_memory = manager.list([[0.0] * x_noise_shape[1]] * x_noise_shape[0])
        u_noise_shared_memory = manager.list([[[0.0] for _ in range(u_noise_shape[1])] for _ in range(u_noise_shape[0])])
        j_noise_shared_memory = manager.list([0.0] * j_noise_shape)
        
        argument_each_group = []
        for idx_ini_guess in range(0, INITIAL_GUESS_NUM): 
            for turn in range(0,num_datagroup):
                # initial guess
                x_ini_guess = initial_guess_x[idx_ini_guess]
                u_ini_guess = initial_guess_u[idx_ini_guess]
                idx_group_of_control_step = idx_ini_guess*num_datagroup+turn
                
                #initial states
                x_0 = rng0[turn,IDX_X_INI]
                theta_0 = rng0[turn,IDX_THETA_INI]
                theta_red_0 = ThetaToRedTheta(theta_0)
                x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
                
                argument_each_group.append((x_ini_guess, u_ini_guess, idx_group_of_control_step, x0, 
                                            x_normal_shared_memory, u_normal_shared_memory, j_normal_shared_memory,
                                            x_noise_shared_memory, u_noise_shared_memory, j_noise_shared_memory))
   
        with Pool(processes=MAX_CORE_CPU) as pool:
            pool.starmap(RunMPCForSingle_IniState_IniGuess, argument_each_group)
                
        # shared memory with manager list
        x_all_normal = torch.from_numpy(np.array(x_normal_shared_memory))
        u_all_normal = torch.from_numpy(np.array(u_normal_shared_memory))
        j_all_normal = torch.from_numpy(np.array(j_normal_shared_memory))
        
        x_all_noisy = torch.from_numpy(np.array(x_noise_shared_memory))
        u_all_noisy = torch.from_numpy(np.array(u_noise_shared_memory))
        j_all_noisy = torch.from_numpy(np.array(j_noise_shared_memory))

        # show the first saved u and x0
        print(f'first_u -- {u_all_normal[0,:,0]}')
        print(f'first_x0 -- {x_all_normal[0,:]}')

        ##### data combing #####
        # u combine u_normal + u_noisy
        u_training_data = torch.cat((u_all_normal, u_all_noisy), dim=0)
        print(f'u_training_data -- {u_training_data.size()}')

        # x0 combine x_normal + x_noisy
        x0_conditioning_data = torch.cat((x_all_normal, x_all_noisy), dim=0)
        print(f'x0_conditioning_data -- {x0_conditioning_data.size()}')

        # J combine j_normal + j_noisy
        J_training_data = torch.cat((j_all_normal, j_all_noisy), dim=0)

        # data saving
        torch.save(u_training_data, os.path.join(SAVE_PATH, U_DATA_NAME))
        torch.save(x0_conditioning_data, os.path.join(SAVE_PATH, X0_CONDITION_DATA_NAME))
        torch.save(J_training_data, os.path.join(SAVE_PATH, J_DATA_NAME))

    # end_time = time.time()

    # duration = end_time - start_time
    # print(f"Time taken for generating data: {duration} seconds")



def main():
    # num_seed = NUM_SEED
    ini_data_start_idx = 0
    # noisy_data_start_idx = NUM_INI_STATES*CONTROL_STEPS

    # initial data generating 50*7, 50*7, 50
    # ini_0_states, random_ini_u_guess, ini_data_idx = ini_data_generating()

    # memories u,x,j


    # memories for data
    u_ini_memory = np.zeros((1*CONTROL_STEPS, HOR, 1)) 
    u_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, HOR, 1))
    x_ini_memory = np.zeros((1*CONTROL_STEPS, 5)) 
    x_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, 5)) 
    j_ini_memory = np.zeros((1*CONTROL_STEPS, 1)) 
    j_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, 1)) 

    # initial data groups 50
    argument_each_group = []
    for idx_ini_guess in range(0, INITIAL_GUESS_NUM): 
        for turn in range(0,num_datagroup):
            # initial guess
            x_ini_guess = initial_guess_x[idx_ini_guess]
            u_ini_guess = initial_guess_u[idx_ini_guess]
            idx_group_of_control_step = idx_ini_guess*num_datagroup+turn
            
            #initial states
            x_0 = rng0[turn,IDX_X_INI]
            theta_0 = rng0[turn,IDX_THETA_INI]
            theta_red_0 = ThetaToRedTheta(theta_0)
            x0 = np.array([x_0, 0.0, theta_0, 0, theta_red_0])
            
            argument_each_group.append((x_ini_guess, u_ini_guess, idx_group_of_control_step, x0, 
                                        u_ini_memory, x_ini_memory, j_ini_memory, u_random_memory, x_random_memory, j_random_memory))
        

    # test_data_group = initial_data_groups[0:2]
    
    # (noisy)
    # for a in range(NUM_INI_STATES):
    #       for b in range(NOISE_DATA_PER_STATE):
    #             initial_data_groups.append([ini_noisy_data_u_guess[a,b,:], ini_noisy_states[a,b,:]])
    
    #     ini_data_groups_array = np.array(initial_data_groups)
    #     print(f'initial_data_groups_size -- {ini_data_groups_array.shape}')

    with Pool(processes=MAX_CORE_CPU) as pool:
          pool.starmap(RunMPCForSingle_IniState_IniGuess, argument_each_group)





if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    main()