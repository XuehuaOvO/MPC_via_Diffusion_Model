import torch
import numpy as np
import os

FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance' 
FILE_NAME = 'cost_.npy'

SAVING_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot' 

# npy load
npy_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/cost_collecting_sample_1/cost_.npy'
npy_data = np.load(npy_path)
print(f'cost -- {npy_data}')

NUMBER = 10

# diffusion cost
diffusion_cost_array = np.zeros((NUMBER,1))

for idx in range(1,11):
    folder_name = 'cost_collecting_sample_' + str(idx)
    folder_path = os.path.join(FOLDER_PATH , folder_name)
    file_path = os.path.join(folder_path , FILE_NAME)
    cost = np.load(file_path)
    diffusion_cost_array[idx-1,0] = cost

diffusion_cost_file_path = os.path.join(FOLDER_PATH, "concatenated_diffusion_cost.npy")
print(f'diffusion size -- {diffusion_cost_array.shape}')
np.save(diffusion_cost_file_path, diffusion_cost_array)


# nmpc cost
NMPC_DATA_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data'

nmpc_cost_array = np.zeros((NUMBER,1))

for i in range(0,10):
    file_name = 'cost_' + 'idx-' + str(i) + '.npy'
    nmpc_file_path = os.path.join(NMPC_DATA_PATH , file_name)
    nmpc_cost = np.load(nmpc_file_path)
    nmpc_cost_array[i,0] = nmpc_cost

nmpc_cost_file_path = os.path.join(FOLDER_PATH, "concatenated_nmpc_cost.npy")
print(f'nmpc size -- {nmpc_cost_array.shape}')
np.save(nmpc_cost_file_path, nmpc_cost_array)