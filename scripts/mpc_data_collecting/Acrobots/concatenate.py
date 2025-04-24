import torch
import numpy as np
import os

FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/mpc_data_collecting/Acrobots/figure/collecting_multi_guess' 

# npy load
# npy_path = '/root/diffusion_mujoco_panda/collecting_test/collecting_6/new_cost_idx-103.npy'
# npy_data = np.load(npy_path)
# print(f'cost -- {npy_data}')

# npy_path = '/root/diffusion_mujoco_panda/collecting_test/collecting_6/time_mpc_idx-102.npy'
# data = np.load(npy_path)
# print(f'delta time -- {data}')

# tensor load
# file_path = "/root/diffusion_mujoco_panda/collecting_test/collecting_6/j_data_idx-101_test6.pt"
# data = torch.load(file_path)
# print(f'delta time -- {data}')

# # tensor size
# print(f'data_size -- {data.size()}')
# print(data[4200,:])

data_idxs = [0, 1]
# data_idxs = np.linspace(0, 49, 50).astype(int)
# to_remove = np.array([4, 5, 6, 14, 16, 34, 35, 45])
# data_idxs = np.setdiff1d(data_idxs, to_remove)

for idx in data_idxs:
    print(f'idx -- {idx}')

tensor_u_list = []
tensor_x_list = []
tensor_j_list = []

# u data cat
for idx in data_idxs:
    file_name = 'u_data_' + 'idx-' + str(idx) + '_MG.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_u_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'u_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_u_list.append(tensor)

concatenated_u_tensor = torch.cat(tensor_u_list, dim=0)
print(f'u size -- {concatenated_u_tensor.shape}')
torch.save(concatenated_u_tensor, os.path.join(FOLDER_PATH , f'u_MG_' + 'cat' + '_400.pt'))

# x data cat
for idx in data_idxs:
    file_name = 'x_data_' + 'idx-' + str(idx) + '_MG.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_x_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'x_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_x_list.append(tensor)

concatenated_x_tensor = torch.cat(tensor_x_list, dim=0)
print(f'x size -- {concatenated_x_tensor.shape}')
torch.save(concatenated_x_tensor, os.path.join(FOLDER_PATH , f'x_MG_' + 'cat' + '_400.pt'))

# j data cat
for idx in data_idxs:
    file_name = 'j_data_' + 'idx-' + str(idx) + '_MG.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_j_list.append(tensor)

# for idx in range(5,28):
#     file_name = 'j_data_' + 'idx-' + str(idx) + '_test5.pt'
#     file_path = os.path.join(FOLDER_PATH , file_name)
#     tensor = torch.load(file_path)
#     tensor_j_list.append(tensor)

concatenated_j_tensor = torch.cat(tensor_j_list, dim=0)
print(f'j size -- {concatenated_j_tensor.shape}')
torch.save(concatenated_j_tensor, os.path.join(FOLDER_PATH , f'j_MG_' + 'cat' + '_400.pt'))