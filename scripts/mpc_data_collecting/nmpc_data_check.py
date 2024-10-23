import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# # data saving folder
u_folder_path = 'cart_pole_diffusion_based_on_MPD/training_data/diff_mpc_2024/u_ini_10x15_noise_15_step_50_hor_64.pt'
u_nmpc = torch.load(u_folder_path)
print(f'u_nmpc -- {u_nmpc.size()}')# 672000*64*1

x_folder_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/diff_mpc_2024/x0_ini_10x15_noise_15_step_50_hor_64.pt'
x_nmpc = torch.load(x_folder_path)
print(f'x_nmpc -- {x_nmpc.size()}') # 672000*5

############## u original training data ##############
u_orig_data_pos = u_nmpc[0:16000,:,:] # 16000*64*1
u_orig_data_neg = u_nmpc[16000:32000,:,:] # 16000*64*1

############## x original training data ##############
x_orig_data_pos = x_nmpc[0:16000,:] # 16000*5
x_orig_data_neg = x_nmpc[16000:32000,:] # 16000*5

############## u noisy training data ##############
u_noi_data_pos = u_nmpc[32000:352000,:,:]
u_noi_data_neg = u_nmpc[352000:672000,:,:]

############## x noisy training data ##############
x_noi_data_pos = x_nmpc[32000:352000,:]
x_nos_data_neg = x_nmpc[352000:672000,:]

############## for the first initial state ##############

############## positive group ##############
######### original #########
pos_orig_x_data = x_orig_data_pos[0:80,:]
pos_orig_u_data = u_orig_data_pos[0:80,:,:]

######### noisy #########
u_noi_data_pos_first = u_noi_data_pos[0:1600,0,0]
u_noi_data_pos_first_reshape = u_noi_data_pos_first.view(80, 20)

x_noi_data_pos_first = x_noi_data_pos [0:1600,:]
# position
x_noi_data_pos_first_position = x_noi_data_pos_first[:,0]
x_noi_data_pos_first_position_reshape = x_noi_data_pos_first_position.view(80, 20)
# velocity
x_noi_data_pos_first_velocity = x_noi_data_pos_first[:,1]
x_noi_data_pos_first_velocity_reshape = x_noi_data_pos_first_velocity.view(80, 20)
# theta
x_noi_data_pos_first_theta = x_noi_data_pos_first[:,2]
x_noi_data_pos_first_theta_reshape = x_noi_data_pos_first_theta.view(80, 20)
# theta_dot
x_noi_data_pos_first_theta_dot = x_noi_data_pos_first[:,3]
x_noi_data_pos_first_theta_dot_reshape = x_noi_data_pos_first_theta_dot.view(80, 20)
# theta_star
x_noi_data_pos_first_theta_star = x_noi_data_pos_first[:,4]
x_noi_data_pos_first_theta_star_reshape = x_noi_data_pos_first_theta_star.view(80, 20)


############## negative group ##############
######### original #########
neg_orig_x_data = x_orig_data_neg[0:80,:]
neg_orig_u_data = u_orig_data_neg[0:80,:,:]

######### noisy #########
u_noi_data_neg_first = u_noi_data_neg[0:1600,0,0]
u_noi_data_neg_first_reshape = u_noi_data_neg_first.view(80, 20)

x_noi_data_neg_first = x_nos_data_neg[0:1600,:]
# position
x_noi_data_neg_first_position = x_noi_data_neg_first[:,0]
x_noi_data_neg_first_position_reshape = x_noi_data_neg_first_position.view(80, 20)
# velocity
x_noi_data_neg_first_velocity = x_noi_data_neg_first[:,1]
x_noi_data_neg_first_velocity_reshape = x_noi_data_neg_first_velocity .view(80, 20)
# theta
x_noi_data_neg_first_theta = x_noi_data_neg_first[:,2]
x_noi_data_neg_first_theta_reshape = x_noi_data_neg_first_theta.view(80, 20)
# theta_dot
x_noi_data_neg_first_theta_dot = x_noi_data_neg_first[:,3]
x_noi_data_neg_first_theta_dot_reshape = x_noi_data_neg_first_theta_dot.view(80, 20)
# theta_star
x_noi_data_neg_first_theta_star = x_noi_data_neg_first[:,4]
x_noi_data_neg_first_theta_star_reshape = x_noi_data_neg_first_theta_star.view(80, 20)



###### Plot ######
num_i = 80
step = np.linspace(0,num_i-1,num_i)
step_u = np.linspace(0,num_i-1,num_i)



plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(step, x_noi_data_pos_first_position_reshape[:,:])
plt.plot(step, pos_orig_x_data[0:80, 0])
plt.plot(step, x_noi_data_neg_first_position_reshape[:,:])
plt.plot(step, neg_orig_x_data[0:80, 0])
# plt.legend(['noise 1','MPC']) 
plt.ylabel('position (m)')
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(step, x_noi_data_pos_first_velocity_reshape[:,:])
plt.plot(step, pos_orig_x_data[0:80, 1])
plt.plot(step, x_noi_data_neg_first_velocity_reshape[:,:])
plt.plot(step, neg_orig_x_data[0:80, 1])
plt.ylabel('velocity (m/s)')
plt.grid()

plt.subplot(6, 1, 3)
plt.plot(step, x_noi_data_pos_first_theta_reshape[:,:])
plt.plot(step, pos_orig_x_data[0:80, 2])
plt.plot(step, x_noi_data_neg_first_theta_reshape[:,:])
plt.plot(step, neg_orig_x_data[0:80, 2])
plt.ylabel('theta (rad)')
plt.grid()

plt.subplot(6, 1, 4)
plt.plot(step, x_noi_data_pos_first_theta_dot_reshape[:,:])
plt.plot(step, pos_orig_x_data[0:80, 3])
plt.plot(step, x_noi_data_neg_first_theta_dot_reshape[:,:])
plt.plot(step, neg_orig_x_data[0:80, 3])
plt.ylabel('theta_dot (rad/s)')
plt.grid()

plt.subplot(6, 1, 5)
plt.plot(step, x_noi_data_pos_first_theta_star_reshape[:,:])
plt.plot(step, pos_orig_x_data[0:80, 4])
plt.plot(step, x_noi_data_neg_first_theta_star_reshape[:,:])
plt.plot(step, neg_orig_x_data[0:80, 4])
plt.ylabel('theta star (rad)')
plt.grid()

plt.subplot(6, 1, 6)
plt.plot(step_u, u_noi_data_pos_first_reshape[:,:])
plt.plot(step_u, pos_orig_u_data[0:80,0,0])
plt.plot(step_u, u_noi_data_neg_first_reshape[:,:])
plt.plot(step_u, neg_orig_u_data[0:80,0,0])
plt.ylabel('Ctl Input (N)')
plt.xlabel('Control Step')
plt.grid()
# plt.show()

# save figure 
results_dir = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/mpc_data_collecting'
figure_name = '2_guess_test.png'
figure_path = os.path.join(results_dir, figure_name)
plt.savefig(figure_path)
