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

# print(torch.__version__)

# rng_x = np.linspace(-1,1,50) # 50 x_0 samples
# rng_theta = np.linspace(-np.pi/4,np.pi/4,50) # 50 theta_0 samples
    
# rng0 = []
# for m in rng_x:
#     for n in rng_theta:
#         rng0.append([m,n])
# rng0 = np.array(rng0,dtype=float)

# # test = 101
# test = 100

# x_0 = rng0[test,0]
# x_0= round(x_0, 3)
# theta_0 = rng0[test,1]
# theta_0= round(theta_0, 3)

# #initial states 
# x0 = np.array([x_0 , 0, theta_0, 0])  # Initial states
# K = np.array([-3.16227764, -3.88839436, 23.42934742, 4.67212194])
# u_start = -np.dot(K, x0)
# print(u_start)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create a large tensor
# x = torch.randn(10000, 10000, device=device)

# # Perform a matrix multiplication
# y = torch.matmul(x, x)

# print("Operation completed on:", y.device)

# u = torch.randn((1, 8, 1))
# print(u)
# for i in range(u.size(1)-1):
#     print(i)
# print(u[:,-1,:])

# x_norm = torch.abs(x)
# print(torch.abs(x))
# print(torch.sum(x_norm))
# rng_x = np.linspace(-1,1,50) # 50 x_0 samples
# rng_theta = np.linspace(-np.pi/4,np.pi/4,50) # 50 theta_0 samples
    
# rng0 = []
# for m in rng_x:
#     for n in rng_theta:
#         rng0.append([m,n])
# rng0 = np.array(rng0,dtype=float)

# # test = 101
# test = 101

# x_0 = rng0[test,0]
# x_0= round(x_0, 3)
# theta_0 = rng0[test,1]
# theta_0= round(theta_0, 3)

# x0 = np.array([[x_0 , 0, theta_0, 0]]) 
# print(x0[0,3])

# text_condition_hidden_dims = (512, 256)
# print(text_condition_hidden_dims)


# # context = torch.rand(6,4)
# # print(context.size())
# # mask_shape = torch.rand(context.size(0),1)
# # # print(mask_shape)
# # mask_size = torch.zeros_like(mask_shape)
# # print(mask_size)

POSITION_INITIAL_RANGE = np.linspace(-1,1,5) 
THETA_INITIAL_RANGE = np.linspace(-np.pi/4,np.pi/4,5) 

rng_x = POSITION_INITIAL_RANGE # 20 x_0 samples
rng_theta = THETA_INITIAL_RANGE # 20 theta_0 samples

rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0,dtype=float)
print(f'initial range -- {rng0}')



# load initial starting state x0
rng_x = np.linspace(-1,1,10) # 10 x_0 samples
rng_theta = np.linspace(-np.pi/4,np.pi/4,10) # 10 theta_0 samples

# all possible initial states combinations
rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0,dtype=float)

# one initial state for test
test = 64                                                                            # ++++++++++++++++++ test_num

x_0 = rng0[test,0]
x_0= round(x_0, 3)
theta_0 = rng0[test,1]
theta_0= round(theta_0, 3)


#initial context
x0 = np.array([[x_0 , 0, theta_0, 0]])  # np.array([[x_0 , 0, theta_0, 0]])  
print(f'x0 -- {x0}')