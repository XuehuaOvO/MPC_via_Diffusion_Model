import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt

############### Seetings #######################

# data saving folder
folder_path = "/home/xiao/mpd-public/scripts/mpc_data_collecting"

# simulation time
T = 3.1  # Total time (seconds) 6.5
dt = 0.1  # Time step (seconds)
t = np.arange(0, T, dt) # time intervals 31
print(t.shape)

N = 8 # prediction horizon

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
rng_x = np.linspace(-3,3,2) 
rng_x_dot = np.linspace(-3,3,2)   
rng_theta = np.linspace(-1,1,1)
rng_theta_dot = np.linspace(-np.pi,np.pi,2) 
rng0 = []
for m in rng_x:
    for n in rng_x_dot:
        for p in rng_theta:
            for q in rng_theta_dot:
                rng0.append([m,n,p,q])
rng0 = np.array(rng0)
num_datagroup = len(rng0)
print(f'rng0 -- {rng0.shape}')

# ##### data collecting loop #####

# data set for each turn
x_track = np.zeros((4, len(t)))
x_predicted_track = np.zeros((num_datagroup*(len(t)-1), N+1, 4))
u_track = np.zeros((1, len(t)-1))

grp_size = len(t)-1

# data (x,u) collecting (saved in PT file)
x_all_tensor = torch.zeros(num_datagroup*grp_size,4) # x0: 2400,000*4
x_predicted_tensor = torch.zeros(num_datagroup*grp_size,N+1,4) # 2400,000*9*4
u_all_tensor = torch.zeros(num_datagroup*grp_size,N,1) # u: 2400,000*8*1


for turn in range(0,num_datagroup):

  num_turn = turn + 1
  num_turn_float = str(num_turn)

  x_0 = rng0[turn,0]
  x_0= round(x_0, 4)
  x_0_dot = rng0[turn,1]
  x_0_dot = round(x_0_dot,4)
  theta_0 = rng0[turn,2]
  theta_0= round(theta_0, 4)
  theta_0_dot = rng0[turn,3]
  theta_0_dot= round(theta_0_dot, 4)
  
  #save the initial states
  x0 = np.array([x_0, x_0_dot, theta_0, theta_0_dot])  # Initial states
  print(f'x0-- {x0}')
  x_track[:,0] = x0

  for i in range(0, len(t)-1):
       # casadi_Opti
       optimizer = ca.Opti()
  
       # x and u mpc prediction along N
       X_pre = optimizer.variable(4, N + 1) 
       print(X_pre)
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
       print(f'X_sol_shape -- {X_sol.shape}')
       x_predicted_track[turn*grp_size+i,:,:] = X_sol.T
       U_sol = sol.value(U_pre)
       print(f'U_sol - {U_sol}')
       
       # select the first updated states as new starting state ans save in the x_track
       x0 = X_sol[:,1]
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
       u_all_tensor[turn*grp_size+i,:,0] = u_tensor


  # save states in tensor
  x_save = x_track[:,:-1]
  x_reshape = np.transpose(x_save)
  x_tensor = torch.tensor(x_reshape)
  x_all_tensor[turn*grp_size:turn*grp_size+grp_size,:] = x_tensor 


#   # plot some results
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
torch.save(u_all_tensor, os.path.join(folder_path, f'u-tensor_240-8-1.pt'))

# save x0 data in PT file as conditional info in training
torch.save(x_all_tensor, os.path.join(folder_path, f'x0-tensor_240-4.pt'))

# save x_predicted data in PT file for possible cost calculation
torch.save(x_predicted_tensor, os.path.join(folder_path, f'x_predicted_tensor_240-9-4.pt'))