import casadi as ca
import control
import numpy as np
import os

import matplotlib.pyplot as plt
import torch


DATASET = 180000
MODEL_NUM = 180000
DATA_SAVED_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/nn_180000/model_nn_180000/x0_18'
X0_IDX = 'interpolated_18'

TRAINING_MODEL_TYPE = 'NN' # Diffusion or NN

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


######### diffusion and mpc states total costs ###########

# diffusion total state cost along the control steps
diffusion_total_state_cost = np.sum(diffusion_x_cost_array, axis = 0)

# mpc total state cost along the control steps
mpc_total_state_cost = np.sum(mpc_x_cost_array, axis = 0)


############# states total costs difference ##############

# difference along the control steps 
state_cost_difference = diffusion_total_state_cost - mpc_total_state_cost 


######### Plot ###########

num_i = data_length
step = np.linspace(0,num_i-1,num_i)
step_u = np.linspace(0,num_i-1,num_i)

plt.figure(figsize=(10, 12))

plt.subplot(7, 1, 1)
plt.plot(step, diffusion_x_cost_array[0, :])
plt.plot(step, mpc_x_cost_array[0, :])
plt.legend([TRAINING_MODEL_TYPE + ' Sampling', 'MPC']) 
plt.ylabel('Position Cost')
plt.title( TRAINING_MODEL_TYPE + '& MPC Cost Plots')
plt.grid()

plt.subplot(7, 1, 2)
plt.plot(step, diffusion_x_cost_array[1, :])
plt.plot(step, mpc_x_cost_array[1, :])
plt.ylabel('Velocity Cost')
plt.grid()

plt.subplot(7, 1, 3)
plt.plot(step, diffusion_x_cost_array[2, :])
plt.plot(step, mpc_x_cost_array[2, :])
plt.ylabel('Angle Cost')
plt.grid()

plt.subplot(7, 1, 4)
plt.plot(step, diffusion_x_cost_array[3, :])
plt.plot(step, mpc_x_cost_array[3, :])
plt.ylabel('Ag Velocity Cost')
plt.grid()

plt.subplot(7, 1, 5)
plt.plot(step, diffusion_total_state_cost.reshape(data_length,))
plt.plot(step, mpc_total_state_cost.reshape(data_length,))
plt.ylabel('Total State Cost')
plt.grid()

plt.subplot(7, 1, 6)
plt.plot(step, state_cost_difference.reshape(data_length,))
plt.ylabel('Cost Difference ('+ TRAINING_MODEL_TYPE +'-MPC)') # Total State Cost Difference
plt.grid()

plt.subplot(7, 1, 7)
plt.plot(step_u, diffusion_u_cost_array.reshape(data_length,))
plt.plot(step_u, mpc_u_cost_array.reshape(data_length,))
plt.ylabel('Ctl Input Cost')
plt.xlabel('Control Step')
plt.grid()
# plt.show()


# save figure 
figure_name = TRAINING_MODEL_TYPE + '_set_' + str(DATASET) + '_model_' + str(MODEL_NUM) + 'x0_' + str(X0_IDX) + '.png'
figure_path = os.path.join(DATA_SAVED_PATH, figure_name)
plt.savefig(figure_path)





