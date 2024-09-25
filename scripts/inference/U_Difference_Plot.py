import casadi as ca
import control
import numpy as np
import os

import matplotlib.pyplot as plt
import torch


DATASET = 2406400
U_SAVED_PATH = '/root/cartpoleDiff/cartpole_inference_u_results'
X0_IDX = 64

################# u results loading ###################

loaded_idx = [10000,30000,50000,70000]

for i in range(1,23):
    model_inx = 100000*i
    loaded_idx.append(model_inx)

print(f'loaded_model_idx -- {loaded_idx}')

u_mes_save = np.zeros((1, len(loaded_idx)))
u_horizon_mse_save = np.zeros((1, len(loaded_idx)))

m = 0

for n in loaded_idx:
    loading_folder = os.path.join(U_SAVED_PATH, 'model_'+ str(n), 'x0_'+ str(X0_IDX))

    u_diffusion_path = os.path.join(loading_folder, 'u_diffusion.npy')
    u_diffusion = np.load(u_diffusion_path)

    u_mpc_path = os.path.join(loading_folder, 'u_mpc.npy')
    u_mpc = np.load(u_mpc_path)

    u_diffusion_horizon_path = os.path.join(loading_folder, 'u_horizon_diffusion.npy')
    u_diffusion_horizon = np.load(u_diffusion_horizon_path)

    u_mpc_horizon_path = os.path.join(loading_folder, 'u_horizon_mpc.npy')
    u_mpc_horizon = np.load(u_mpc_horizon_path)
    
    # calculate mse
    u_mse = np.mean((u_diffusion - u_mpc) ** 2)
    u_horizon_mse = np.mean((u_diffusion_horizon - u_mpc_horizon) ** 2)

    # save mse
    u_mes_save[0,m] = u_mse
    u_horizon_mse_save[0,m] = u_horizon_mse

    m += 1
    print(f'm -- {m}')


################# Plotting ###################

plt.figure(figsize=(10, 10))

plt.plot(loaded_idx, u_mes_save[0, :])
plt.plot(loaded_idx, u_horizon_mse_save[0, :])
plt.yscale('log')
plt.legend(['u_mes', 'u_horizon_mse']) 
plt.ylabel('MSE value')
plt.xlabel('Iterations')
plt.grid()

# save figure 
plot_name = 'dataset_' + str(DATASET) + '_x0_' + str(X0_IDX) + '_Diffusion_MPC_MSE' + '.png'
plot_path = os.path.join(U_SAVED_PATH, plot_name)
plt.savefig(plot_path)




