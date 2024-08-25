import torch
import random
import numpy as np
import torch.nn.functional as Functional
import os

# Get the number of CPU cores
# cpu_count = os.cpu_count()
# print(f"Number of CPU cores: {cpu_count}")

# if torch.cuda.is_available():
#     print("CUDA is available. GPU can be used.")
#     print(f"Device Name: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available. GPU cannot be used.")

# Create a tensor and move it to the GPU
# x = torch.randn(10000, 10000).cuda()
# y = torch.matmul(x, x)

# print(f"Result of matrix multiplication: {y}")

# q_limits=torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
# q_min = q_limits[0]
# print(q_min)
# q_max= q_limits[1]
# print(q_max)
# q_distribution = torch.distributions.uniform.Uniform(q_min, q_max)
# q_pos = q_distribution.sample((10,))
# # print(q_pos)
# q_dim = len(q_min)
# x_pos = q_pos[...,:q_dim]
# # print(x_pos)
# # print(x_pos.ndim)
# # print(x_pos.shape)
# # print(x_pos.unsqueeze(1).shape)
# b = x_pos.shape[0]
# collisions = torch.ones((b, 1)) 
# # print(collisions)
# idxs_not_in_collision = torch.argwhere(collisions == False).squeeze()
# # print(collisions == False)
# # print(idxs_not_in_collision)
# idx_random = torch.randperm(len(idxs_not_in_collision))[:1]
# # print(idx_random)
# free_qs = q_pos[idxs_not_in_collision[idx_random]]
# print(free_qs)
# samples = torch.zeros((1, 2))
# idx_end = min(0 + free_qs.shape[0], samples.shape[0])
# print(idx_end)
# samples[0:idx_end] = free_qs[:idx_end - 0]
# link_names_for_object_collision_checking=['link_0', 'check', 'element']
# print(len(link_names_for_object_collision_checking))

# x = torch.rand(4,3,3)
# x_select = x [..., [1]]
# print(x)
# print(x_select)

# # Example tensors representing points in some space
# q1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# q2 = torch.tensor([[4.0, 6.0], [5.0, 9.0]])

# q = q1 - q2
# print(q)
# # Assume self is defined; directly calling the function
# distance = torch.linalg.norm(q, dim=1)

# print(distance)


# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     print(out)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# a = torch.tensor([0.9, 0.8, 0.9, 0.5, 0.6, 0.7])  
# print(a.shape)
# t = torch.tensor([0, 1]) 
# print(t.shape)
# x_shape = (2, 4, 4) 
# print(x_shape)

# out = extract(a, t, x_shape)
# print(out.shape)
# print(len((2,5)))

# random.seed(30)
# np.random.seed(30)
# print(np.random.rand(4)) 

# t = torch.tensor([1, 2, 3])
# print(t.shape)
# print(t[1])

# points = torch.randn(2,3,4)
# trans_points = points.transpose(-2,-1)
# print(trans_points.shape)
# interpolate = Functional.interpolate(points.transpose(-2, -1), size=10, mode='linear', align_corners=True)
# print(interpolate.shape)

# task = torch.zeros(1, 1)
# print(task.shape[0])

print(torch.__version__)