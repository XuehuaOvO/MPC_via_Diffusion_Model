import casadi as ca
import control
import numpy as np
import os

import matplotlib.pyplot as plt
import torch


DATASET = 420000
MODEL_NUM = 230000
DATA_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/420000set/model_230000/x0_150'
X0_IDX = 150

################# performance results loading ###################

# diffusion results
diffusion_u_result_path = os.path.join(DATA_SAVED_PATH, 'u_horizon_diffusion.npy')
diffusion_u_result = np.load(diffusion_u_result_path) # 50*8

diffusion_x_result_path = os.path.join(DATA_SAVED_PATH, 'x_diffusion_horizon.npy')
diffusion_x_result = np.load(diffusion_x_result_path) # 50*9*4

# mpc results
mpc_u_result_path = os.path.join(DATA_SAVED_PATH, 'u_horizon_mpc.npy')
mpc_u_result = np.load(mpc_u_result_path) # 50*8

mpc_x_result_path = os.path.join(DATA_SAVED_PATH, 'x_mpc_horizon.npy')
mpc_x_result = np.load(mpc_x_result_path) # 50*9*4

################# results cost calculating (Weight = 1) ###################
# steady_x = np.array([0, 0, 0, 0])
# steady_u = 0

data_length = diffusion_u_result.shape[0]

diffusion_u_cost_array = np.zeros((1, data_length))
mpc_u_cost_array = np.zeros((1, data_length))

diffusion_x_cost_array = np.zeros((4, data_length))
mpc_x_cost_array = np.zeros((4, data_length))

# diffusion u cost
for i in range(0,data_length):
    single_diffusion_u_cost = (diffusion_u_result[i,:])**2
    diffusion_u_cost_array[0,i] = np.sum(single_diffusion_u_cost)

# mpc u cost
for i in range(0,data_length):
    single_mpc_u_cost = (mpc_u_result[i,:])**2
    mpc_u_cost_array[0,i] = np.sum(single_mpc_u_cost)

# diffusion x cost
for i in range(0,data_length):
    single_diffusion_x_cost = (diffusion_x_result[i,:,:])**2
    diffusion_x_cost_array[:,i] = np.sum(single_diffusion_x_cost,axis=0)

# mpc x cost
for i in range(0,data_length):
    single_mpc_x_cost = (mpc_x_result[i,:,:])**2
    mpc_x_cost_array[:,i] = np.sum(single_mpc_x_cost,axis=0)


######### Plot ###########

num_i = data_length
step = np.linspace(0,num_i-1,num_i)
step_u = np.linspace(0,num_i-1,num_i)

plt.figure(figsize=(10, 8))

plt.subplot(5, 1, 1)
plt.plot(step, diffusion_x_cost_array[0, :])
plt.plot(step, mpc_x_cost_array[0, :])
plt.legend(['Diffusion Sampling', 'MPC']) 
plt.ylabel('Position Cost (m)')
plt.grid()

plt.subplot(5, 1, 2)
plt.plot(step, diffusion_x_cost_array[1, :])
plt.plot(step, mpc_x_cost_array[1, :])
plt.ylabel('Velocity_Cost (m/s)')
plt.grid()

plt.subplot(5, 1, 3)
plt.plot(step, diffusion_x_cost_array[2, :])
plt.plot(step, mpc_x_cost_array[2, :])
plt.ylabel('Angle_cost (rad)')
plt.grid()

plt.subplot(5, 1, 4)
plt.plot(step, diffusion_x_cost_array[3, :])
plt.plot(step, mpc_x_cost_array[3, :])
plt.ylabel('Ag Velocity_Cost (rad/s)')
plt.grid()

plt.subplot(5, 1, 5)
plt.plot(step_u, diffusion_u_cost_array.reshape(data_length,))
plt.plot(step_u, mpc_u_cost_array.reshape(data_length,))
plt.ylabel('Ctl Input (N)')
plt.xlabel('Control Step')
plt.grid()
# plt.show()
# save figure 
figure_name = 'set_' + str(DATASET) + '_model_' + str(MODEL_NUM) + 'x0_' + str(X0_IDX) + '.png'
figure_path = os.path.join(DATA_SAVED_PATH, figure_name)
plt.savefig(figure_path)

################# u results loading ###################

# loaded_idx = [10000,30000,50000,70000]

# for i in range(1,23):
#     model_inx = 100000*i
#     loaded_idx.append(model_inx)

# print(f'loaded_model_idx -- {loaded_idx}')

# u_mes_save = np.zeros((1, len(loaded_idx)))
# u_horizon_mse_save = np.zeros((1, len(loaded_idx)))

# m = 0

# for n in loaded_idx:
#     loading_folder = os.path.join(DATA_SAVED_PATH, 'model_'+ str(n), 'x0_'+ str(X0_IDX))

#     u_diffusion_path = os.path.join(loading_folder, 'u_diffusion.npy')
#     u_diffusion = np.load(u_diffusion_path)

#     u_mpc_path = os.path.join(loading_folder, 'u_mpc.npy')
#     u_mpc = np.load(u_mpc_path)

#     u_diffusion_horizon_path = os.path.join(loading_folder, 'u_horizon_diffusion.npy')
#     u_diffusion_horizon = np.load(u_diffusion_horizon_path)

#     u_mpc_horizon_path = os.path.join(loading_folder, 'u_horizon_mpc.npy')
#     u_mpc_horizon = np.load(u_mpc_horizon_path)
    
#     # calculate mse
#     u_mse = np.mean((u_diffusion - u_mpc) ** 2)
#     u_horizon_mse = np.mean((u_diffusion_horizon - u_mpc_horizon) ** 2)

#     # save mse
#     u_mes_save[0,m] = u_mse
#     u_horizon_mse_save[0,m] = u_horizon_mse

#     m += 1
#     print(f'm -- {m}')


# ################# Plotting ###################

# plt.figure(figsize=(10, 10))

# plt.plot(loaded_idx, u_mes_save[0, :])
# plt.plot(loaded_idx, u_horizon_mse_save[0, :])
# plt.yscale('log')
# plt.legend(['u_mes', 'u_horizon_mse']) 
# plt.ylabel('MSE value')
# plt.xlabel('Iterations')
# plt.grid()

# # save figure 
# plot_name = 'dataset_' + str(DATASET) + '_x0_' + str(X0_IDX) + '_Diffusion_MPC_MSE' + '.png'
# plot_path = os.path.join(DATA_SAVED_PATH, plot_name)
# plt.savefig(plot_path)




