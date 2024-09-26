import torch
import os

# data saving folder
folder_path = "/root/cartpoleDiff/cartpole_lmpc_data"

# u loading 
u_2400000 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/u-tensor_2400000-8-1.pt")
u_6400 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/u-tensor_6400-8-1.pt") 

# x0 loading 
x0_2400000 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/x0-tensor_2400000-4.pt") 
x0_6400 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/x0-tensor_6400-4.pt") 

# concatenate
u_tensor_2406400 = torch.cat((u_2400000, u_6400), dim=0)
print(f'u_tensor_2406400 -- {u_tensor_2406400.size()}')

x0_tensor_2406400 = torch.cat((x0_2400000, x0_6400), dim=0)
print(f'x0_tensor_49600 -- {x0_tensor_2406400.size()}')

# save
torch.save(u_tensor_2406400, os.path.join(folder_path, f'u_tensor_2406400-8-1.pt'))
torch.save(x0_tensor_2406400, os.path.join(folder_path, f'x0_tensor_2406400-4.pt'))