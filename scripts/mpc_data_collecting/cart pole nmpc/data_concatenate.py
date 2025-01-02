import torch
import numpy as np
import os

IDX_NUM = 50
FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data_collecting/nmpc_cart_pole_collecting'

# npy load
# npy_path = '/root/diffusion_mujoco_panda/inference_SG_MG/j_SG_idx-6_test.npy'
# npy_data = np.load(npy_path)

# tensor load
# file_path = "/root/diffusion_mujoco_panda/collecting_test/collecting_5/x_data_cat_test5.pt"
# data = torch.load(file_path)

# # tensor size
# print(f'data_size -- {data.size()}')
# print(data[4200,:])

data_idxs = np.arange(0, 50, 1)

for idx in data_idxs:
    print(f'idx -- {idx}')

tensor_u_list = []
tensor_x_list = []
tensor_j_list = []

# u data cat
for idx in data_idxs:
    file_name = 'u_data_' + 'idx-' + str(idx) + '_test1.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_u_list.append(tensor)


concatenated_u_tensor = torch.cat(tensor_u_list, dim=0)
torch.save(concatenated_u_tensor, os.path.join(FOLDER_PATH , f'u_data_' + 'cat' + '_test1.pt'))

# x data cat
for idx in data_idxs:
    file_name = 'x_data_' + 'idx-' + str(idx) + '_test1.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_x_list.append(tensor)


concatenated_x_tensor = torch.cat(tensor_x_list, dim=0)
torch.save(concatenated_x_tensor, os.path.join(FOLDER_PATH , f'x_data_' + 'cat' + '_test1.pt'))

# j data cat
for idx in data_idxs:
    file_name = 'j_data_' + 'idx-' + str(idx) + '_test1.pt'
    file_path = os.path.join(FOLDER_PATH , file_name)
    tensor = torch.load(file_path)
    tensor_j_list.append(tensor)


concatenated_j_tensor = torch.cat(tensor_j_list, dim=0)
torch.save(concatenated_j_tensor, os.path.join(FOLDER_PATH , f'j_data_' + 'cat' + '_test1.pt'))