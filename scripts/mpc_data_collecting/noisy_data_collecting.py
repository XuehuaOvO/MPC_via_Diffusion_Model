import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt

############### Seetings ######################
# Attention: this py file can only set the initial range of position and theta, initial x_dot and theta_dot are always 0

# data saving folder
folder_path = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/CartPole-LMPC"

# control steps
CONTROL_STEPS = 50

# data range
POSITION_INITIAL_RANGE = np.linspace(-1,1,20) 
THETA_INITIAL_RANGE = np.linspace(-np.pi/4,np.pi/4,20) 

# number of noisy data for each state
NUM_NOISY_DATA = 20

N = 8 # mpc prediction horizon

# trainind data files name
U_DATA_NAME = 'noisy_u_all_400000-8-1.pt' # 400000: training data amount, 8: horizon length, 1:channels --> 400000-8-1: tensor size for data trainig 
X0_CONDITION_DATA_NAME = 'noisy_x_all_400000-4.pt' # 400000-4: tensor size for conditioning data in training

np.random.seed(42)

############### Dynamics Define ######################
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



############# MPC #####################

# mpc parameters
Q = np.diag([10, 1, 10, 1]) 
R = np.array([[1]])
P = np.diag([100, 1, 100, 1])

# x_ref = ca.SX.sym('x_ref', 4)

# Define the initial states range
rng_x = POSITION_INITIAL_RANGE 
rng_theta = THETA_INITIAL_RANGE 
rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0)
num_datagroup = len(rng0) # 20*20 = 400
print(f'rng0 -- {rng0.shape}')

# ##### data collecting loop #####

# data set for each turn
x_track = np.zeros((4, (CONTROL_STEPS+1)))
x_predicted_track = np.zeros((num_datagroup*CONTROL_STEPS, N+1, 4))
u_track = np.zeros((1, CONTROL_STEPS))

# data (x,u) collecting (saved in PT file)
x_all_tensor = torch.zeros(num_datagroup*(CONTROL_STEPS),4) # x0: 20000*4
x_predicted_tensor = torch.zeros(num_datagroup*(CONTROL_STEPS),N+1,4) # 20000*9*4
u_all_tensor = torch.zeros(num_datagroup*(CONTROL_STEPS),N,1) # u: 20000*8*1

# all noisy data
num_noisy_state = NUM_NOISY_DATA

noisy_x_all = torch.zeros(num_datagroup*(CONTROL_STEPS)*num_noisy_state, 4) # 400000*4
noisy_u_all = torch.zeros(num_datagroup*(CONTROL_STEPS)*num_noisy_state, N, 1) # 400000*8*1

for turn in range(0,num_datagroup):

  num_turn = turn + 1
  num_turn_float = str(num_turn)

  x_0 = rng0[turn,0]
  x_0= round(x_0, 4)
  theta_0 = rng0[turn,1]
  theta_0= round(theta_0, 4)
  
  #save the initial states
  x0 = np.array([x_0, 0, theta_0, 0])  # Initial states
  print(f'x0-- {x0}')
  x_track[:,0] = x0

  ##### noisey data generating #####

  # noisy x0 
  # num_noisy_state = 20
  noise_mean = 0
  noise_sd = 0.15

  noisey_x_array = np.zeros((num_noisy_state, x0.shape[0]))

  # noisy x0
  for n in range(0,num_noisy_state):
        noise = np.random.normal(noise_mean, noise_sd, size = (1,2))
        noisy_state = x0 + [noise[0,0], 0, noise[0,1],0]
        noisy_state = np.round(noisy_state, 4)
        noisey_x_array [n,:] = noisy_state 

  # save the initail noisy x group
  noisy_x_all[turn*(CONTROL_STEPS)*num_noisy_state:turn*(CONTROL_STEPS)*num_noisy_state+num_noisy_state,:] = torch.tensor(noisey_x_array)
  print(f'start,end -- {turn*(CONTROL_STEPS)*num_noisy_state, turn*(CONTROL_STEPS)*num_noisy_state+num_noisy_state}')
  print(f'noisy_x0 -- {noisy_x_all[turn*(CONTROL_STEPS)*num_noisy_state:turn*(CONTROL_STEPS)*num_noisy_state+num_noisy_state,:]}')

  # main mpc loop
  for i in range(0, CONTROL_STEPS):
       # casadi_Opti
       # optimizer = ca.Opti()

       ################################## u calculating for noisy data ##################################
       u_for_noisy_x = np.zeros((num_noisy_state, N))

       # X_noisy_pre = optimizer.variable(4, N + 1) 
       # U_noisy_pre = optimizer.variable(1, N)

       for m in range(0,len(noisey_x_array)):
           # casadi_Opti
           optimizer = ca.Opti()

           X_noisy_pre = optimizer.variable(4, N + 1) 
           U_noisy_pre = optimizer.variable(1, N)

           optimizer.subject_to(X_noisy_pre[:, 0] == noisey_x_array[m,:])

           cost_noisy = 0

           # initial cost
           cost_noisy += Q[0,0]*X_noisy_pre[0, 0]**2 + Q[1,1]*X_noisy_pre[1, 0]**2 + Q[2,2]*X_noisy_pre[2, 0]**2 + Q[3,3]*X_noisy_pre[3, 0]**2

           # state cost
           for k in range(0,N-1):
               x_noisy_next = cart_pole_dynamics(X_noisy_pre[:, k], U_noisy_pre[:, k])
               optimizer.subject_to(X_noisy_pre[:, k + 1] == x_noisy_next)
               cost_noisy += Q[0,0]*X_noisy_pre[0, k+1]**2 + Q[1,1]*X_noisy_pre[1, k+1]**2 + Q[2,2]*X_noisy_pre[2, k+1]**2 + Q[3,3]*X_noisy_pre[3, k+1]**2 + U_noisy_pre[:, k]**2
            
           # terminal cost
           x_noisy_terminal = cart_pole_dynamics(X_noisy_pre[:, N-1], U_noisy_pre[:, N-1])
           optimizer.subject_to(X_noisy_pre[:, N] == x_noisy_terminal)
           cost_noisy += P[0,0]*X_noisy_pre[0, N]**2 + P[1,1]*X_noisy_pre[1, N]**2 + P[2,2]*X_noisy_pre[2, N]**2 + P[3,3]*X_noisy_pre[3, N]**2 + U_noisy_pre[:, N-1]**2

           optimizer.minimize(cost_noisy)
           optimizer.solver('ipopt')
           usol = optimizer.solve()

           U_noisy_sol = usol.value(U_noisy_pre)
           print(f'turn, i, U_noisy_sol -- {turn, i, U_noisy_sol}')
           
           for v in range(0,len(U_noisy_sol)):
               u_for_noisy_x[m,v] = U_noisy_sol[v]

       # save noisy u 
       noisy_u_all[turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state : turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state + num_noisy_state,:,0] = torch.tensor(u_for_noisy_x)
       print(f'u_start, u_end -- {turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state, turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state + num_noisy_state}')

       ################################################# normal mpc loop to update state #################################################
       
       # casadi_Opti
       optimizer = ca.Opti()
       
       ##### normal mpc #####  
       # x and u mpc prediction along N
       X_pre = optimizer.variable(4, N + 1) 
       U_pre = optimizer.variable(1, N) 

       optimizer.subject_to(X_pre[:, 0] == x0)  # starting state

       # cost 
       cost = 0

       # initial cost
       cost += Q[0,0]*X_pre[0, 0]**2 + Q[1,1]*X_pre[1, 0]**2 + Q[2,2]*X_pre[2, 0]**2 + Q[3,3]*X_pre[3, 0]**2

       # state cost
       for k in range(0,N-1):
          x_next = cart_pole_dynamics(X_pre[:, k], U_pre[:, k])
          optimizer.subject_to(X_pre[:, k + 1] == x_next)
          cost += Q[0,0]*X_pre[0, k+1]**2 + Q[1,1]*X_pre[1, k+1]**2 + Q[2,2]*X_pre[2, k+1]**2 + Q[3,3]*X_pre[3, k+1]**2 + U_pre[:, k]**2

       # terminal cost
       x_terminal = cart_pole_dynamics(X_pre[:, N-1], U_pre[:, N-1])
       optimizer.subject_to(X_pre[:, N] == x_terminal)
       cost += P[0,0]*X_pre[0, N]**2 + P[1,1]*X_pre[1, N]**2 + P[2,2]*X_pre[2, N]**2 + P[3,3]*X_pre[3, N]**2 + U_pre[:, N-1]**2

       optimizer.minimize(cost)
       optimizer.solver('ipopt')
       sol = optimizer.solve()

       X_sol = sol.value(X_pre)
       # print(f'X_sol_shape -- {X_sol.shape}')
       x_predicted_track[turn*(CONTROL_STEPS)+i,:,:] = X_sol.T
       U_sol = sol.value(U_pre)
       print(f'U_sol - {U_sol}') 

       
       # select the first updated states as new starting state ans save in the x_track
       x0 = X_sol[:,1]
       x0 = np.round(x0, 4)
       print(f'x0_new-- {x0}')
       x_track[:,i+1] = x0

       #save the first computed control input
       u_track[:,i] = U_sol[0]
       # print(f'u_track- {u_track}')

       # save control inputs in tensor
       u_reshape = U_sol.reshape(1,N)
       u_tensor = torch.tensor(u_reshape)
       # print(f'u_tensor_shape -- {u_tensor}')
       print(f'turn, i -- {turn, i}')
       u_all_tensor[turn*(CONTROL_STEPS)+i,:,0] = u_tensor

       ################################## noisy data generating ##################################
       noisey_x_array = np.zeros((num_noisy_state, x0.shape[0]))

       for n in range(0,num_noisy_state):
            noise = np.random.normal(noise_mean, noise_sd, size = x0.shape[0])
            noisy_state = x0 + noise
            noisey_x_array [n,:] = noisy_state 

       # print(f'noisey_x_array -- {noisey_x_array}')
         
       # save the initail noisy x group (except the last x0)
       if i != CONTROL_STEPS - 1:
            noisy_x_all[turn*(CONTROL_STEPS)*num_noisy_state + (i+1)*num_noisy_state : turn*(CONTROL_STEPS)*num_noisy_state + (i+1)*num_noisy_state + num_noisy_state,:] = torch.tensor(noisey_x_array)
       print(f'start,end -- {turn*(CONTROL_STEPS)*num_noisy_state + (i+1)*num_noisy_state, turn*(CONTROL_STEPS)*num_noisy_state + (i+1)*num_noisy_state + num_noisy_state}')
       print(f'noisy x last -- {noisy_x_all[turn*(CONTROL_STEPS)*num_noisy_state + (i)*num_noisy_state : turn*(CONTROL_STEPS)*num_noisy_state + (i)*num_noisy_state + num_noisy_state,:]}')
       print(f'noisy u last -- {noisy_u_all[turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state : turn*(CONTROL_STEPS)*num_noisy_state + i*num_noisy_state + num_noisy_state,:,0]}')


  # save states in tensor
  x_save = x_track[:,:-1]
  x_reshape = np.transpose(x_save)
  x_tensor = torch.tensor(x_reshape)
  x_all_tensor[turn*(CONTROL_STEPS):turn*(CONTROL_STEPS)+(CONTROL_STEPS),:] = x_tensor 


  # plot some results
  #   num_i = len(t)-1
  #   step = np.linspace(0,num_i+2,num_i+1)
  #   step_u = np.linspace(0,num_i+1,num_i)

    #   if turn in (0, 32, 64): 
    #      plt.figure(figsize=(10, 8))

    #      plt.subplot(5, 1, 1)
    #      plt.plot(step, x_track[0, :])
    #      plt.ylabel('Position (m)')
    #      plt.grid()

    #      plt.subplot(5, 1, 2)
    #      plt.plot(step, x_track[1, :])
    #      plt.ylabel('Velocity (m/s)')
    #      plt.grid()

    #      plt.subplot(5, 1, 3)
    #      plt.plot(step, x_track[2, :])
    #      plt.ylabel('Angle (rad)')
    #      plt.grid()

    #      plt.subplot(5, 1, 4)
    #      plt.plot(step, x_track[3, :])
    #      plt.ylabel('Ag Velocity (rad/s)')
    #      plt.grid()

    #      plt.subplot(5, 1, 5)
    #      plt.plot(step_u, u_track.reshape(len(t)-1,))
    #      plt.ylabel('Ctl Input (N)')
    #      plt.xlabel('Control Step')
    #      plt.grid()

    #      # save plot 
    #      plotfile = "plt"
    #      plot_name = plotfile + " " + num_turn_float + '.png'
    #      full_plot = os.path.join(folder_path, plot_name)
    #      plt.savefig(full_plot)

x_predicted_tensor = torch.tensor(x_predicted_track)

# show the first saved u and x0
print(f'first_u -- {u_all_tensor[0,:,0]}')
print(f'first_x0 -- {x_all_tensor[0,:]}')
print(f'first_pre_x -- {x_predicted_tensor[0,:,:]}')

# save u data in PT file for training
# torch.save(u_all_tensor, os.path.join(folder_path, f'u-tensor_20000-8-1.pt'))

# # save x0 data in PT file as conditional info in training
# torch.save(x_all_tensor, os.path.join(folder_path, f'x0-tensor_20000-4.pt'))

# # save x_predicted data in PT file for possible cost calculation
# torch.save(x_predicted_tensor, os.path.join(folder_path, f'x_predicted_tensor_20000-9-4.pt'))

# # save noisy_x_all in PT file
# torch.save(noisy_x_all, os.path.join(folder_path, f'noisy_x_all_400000-4.pt'))

# # save u data in PT file for training
# torch.save(noisy_u_all, os.path.join(folder_path, f'noisy_u_all_400000-8-1.pt'))

##### data combing #####

# u combine 
u_training_data = torch.cat((noisy_u_all, u_all_tensor), dim=0)
print(f'u_training_data -- {u_training_data.size()}')

# x0 combine
x0_conditioning_data = torch.cat((noisy_x_all, x_all_tensor), dim=0)
print(f'x0_conditioning_data -- {x0_conditioning_data.size()}')

# data saving
torch.save(u_training_data, os.path.join(folder_path, U_DATA_NAME))
torch.save(x0_conditioning_data, os.path.join(folder_path, X0_CONDITION_DATA_NAME))