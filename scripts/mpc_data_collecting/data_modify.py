import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# data saving folder
folder_path = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/diff_mpc_2024/5_6_noise_3"

# # u loading 
# u_2400000 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/u-tensor_2400000-8-1.pt")
# u_6400 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/u-tensor_6400-8-1.pt") 

# x0 loading 
x0_5 = torch.load("/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/diff_mpc_2024/5_6_noise_3/x0_ini_5x6_noise_3_step_50_hor_64.pt") 
print(f'x0_120000 -- {x0_5.size()}')
# x0_6400 = torch.load("/root/cartpoleDiff/cartpole_lmpc_data/x0-tensor_6400-4.pt") 

# Copy data from the fifth column (index 4) to the third column (index 2)
x0_5[:, 2] = x0_5[:, 4]
# Remove the fifth column (index 4) by slicing
x0_4= x0_5[:, :4]  # This keeps only the first four columns
print(f'x0_4 -- {x0_4.size()}')

# concatenate
# u_tensor_2406400 = torch.cat((u_2400000, u_6400), dim=0)
# print(f'u_tensor_2406400 -- {u_tensor_2406400.size()}')

# x0_tensor_2406400 = torch.cat((x0_2400000, x0_6400), dim=0)
# print(f'x0_tensor_49600 -- {x0_tensor_2406400.size()}')

# save
torch.save(x0_4, os.path.join(folder_path, f'x0_ini_5x6_noise_3_step_50_hor_64_4DoF.pt'))
# torch.save(x0_tensor_2406400, os.path.join(folder_path, f'x0_tensor_2406400-4.pt'))

# u_folder_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/CartPole-LMPC/u_ini_10x20_noise_20_step_80_hor_64.pt'
# u_nmpc = torch.load(u_folder_path)
# print(f'u_nmpc -- {u_nmpc.size()}')# 672000*64*1

# x_folder_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/training_data/CartPole-LMPC/x0_ini_10x20_noise_20_step_80_hor_64.pt'
# x_nmpc = torch.load(x_folder_path)
# print(f'x_nmpc -- {x_nmpc.size()}') # 672000*5

# ############## u original training data ##############
# u_orig_data_pos = u_nmpc[0:16000,:,:] # 16000*64*1
# u_orig_data_neg = u_nmpc[16000:32000,:,:] # 16000*64*1

# ############## x original training data ##############
# x_orig_data_pos = x_nmpc[0:16000,:] # 16000*5
# x_orig_data_neg = x_nmpc[16000:32000,:] # 16000*5

# ############## u noisy training data ##############
# u_noi_data_pos = u_nmpc[32000:352000,:,:]
# u_noi_data_neg = u_nmpc[352000:672000,:,:]

# ############## x noisy training data ##############
# x_noi_data_pos = x_nmpc[32000:352000,:]
# x_nos_data_neg = x_nmpc[352000:672000,:]

# ############## for the first initial state ##############

# ############## positive group ##############
# ######### original #########
# pos_orig_x_data = x_orig_data_pos[0:80,:]
# pos_orig_u_data = u_orig_data_pos[0:80,:,:]

# ######### noisy #########
# u_noi_data_pos_first = u_noi_data_pos[0:1600,0,0]
# u_noi_data_pos_first_reshape = u_noi_data_pos_first.view(80, 20)

# x_noi_data_pos_first = x_noi_data_pos [0:1600,:]
# # position
# x_noi_data_pos_first_position = x_noi_data_pos_first[:,0]
# x_noi_data_pos_first_position_reshape = x_noi_data_pos_first_position.view(80, 20)
# # velocity
# x_noi_data_pos_first_velocity = x_noi_data_pos_first[:,1]
# x_noi_data_pos_first_velocity_reshape = x_noi_data_pos_first_velocity.view(80, 20)
# # theta
# x_noi_data_pos_first_theta = x_noi_data_pos_first[:,2]
# x_noi_data_pos_first_theta_reshape = x_noi_data_pos_first_theta.view(80, 20)
# # theta_dot
# x_noi_data_pos_first_theta_dot = x_noi_data_pos_first[:,3]
# x_noi_data_pos_first_theta_dot_reshape = x_noi_data_pos_first_theta_dot.view(80, 20)
# # theta_star
# x_noi_data_pos_first_theta_star = x_noi_data_pos_first[:,4]
# x_noi_data_pos_first_theta_star_reshape = x_noi_data_pos_first_theta_star.view(80, 20)


# ############## negative group ##############
# ######### original #########
# neg_orig_x_data = x_orig_data_neg[0:80,:]
# neg_orig_u_data = u_orig_data_neg[0:80,:,:]

# ######### noisy #########
# u_noi_data_neg_first = u_noi_data_neg[0:1600,0,0]
# u_noi_data_neg_first_reshape = u_noi_data_neg_first.view(80, 20)

# x_noi_data_neg_first = x_nos_data_neg[0:1600,:]
# # position
# x_noi_data_neg_first_position = x_noi_data_neg_first[:,0]
# x_noi_data_neg_first_position_reshape = x_noi_data_neg_first_position.view(80, 20)
# # velocity
# x_noi_data_neg_first_velocity = x_noi_data_neg_first[:,1]
# x_noi_data_neg_first_velocity_reshape = x_noi_data_neg_first_velocity .view(80, 20)
# # theta
# x_noi_data_neg_first_theta = x_noi_data_neg_first[:,2]
# x_noi_data_neg_first_theta_reshape = x_noi_data_neg_first_theta.view(80, 20)
# # theta_dot
# x_noi_data_neg_first_theta_dot = x_noi_data_neg_first[:,3]
# x_noi_data_neg_first_theta_dot_reshape = x_noi_data_neg_first_theta_dot.view(80, 20)
# # theta_star
# x_noi_data_neg_first_theta_star = x_noi_data_neg_first[:,4]
# x_noi_data_neg_first_theta_star_reshape = x_noi_data_neg_first_theta_star.view(80, 20)



# # positive initial guess part
# u_positive_initial_guess = u_nmpc[0:336000,:,:]
# print(f'u_positive_initial_guess -- {u_positive_initial_guess.size()}')# 336000*8*1
# x_positive_initial_guess = x_nmpc[0:336000,:]
# print(f'x_positive_initial_guess -- {x_positive_initial_guess.size()}')# 336000*8*1

# # negative initial guess part
# u_negative_initial_guess = u_nmpc[336000:,:,:]
# print(f'u_negative_initial_guess -- {u_negative_initial_guess.size()}')# 336000*8*1
# x_negative_initial_guess = x_nmpc[336000:,:]
# print(f'x_negative_initial_guess -- {x_negative_initial_guess.size()}')# 336000*8*1



# ############### positive guess ###############
# # positive initial guess original data
# pos_orig_u_data = u_positive_initial_guess[0:16000,:,:]
# print(f'pos_orig_u_data -- {pos_orig_u_data.size()}') # 16000*64*1
# pos_orig_x_data =x_positive_initial_guess[0:16000,:]
# print(f'pos_orig_x_data -- {pos_orig_x_data.size()}') # 16000*5

# # positive guess noisy data
# pos_noisy_u = u_positive_initial_guess[16000:,:,:]
# print(f'pos_noisy_u -- {pos_noisy_u.size()}') # 320000*64*1
# pos_noisy_x = x_positive_initial_guess[16000:,:]
# print(f'pos_noisy_x -- {pos_noisy_x.size()}') # 320000*5


# ############### negative guess ###############
# # negative initial guess original data
# neg_orig_u_data = u_negative_initial_guess[0:16000,:,:]
# print(f'neg_orig_u_data -- {neg_orig_u_data.size()}')
# neg_orig_x_data =x_negative_initial_guess[0:16000,:]
# print(f'neg_orig_x_data -- {neg_orig_x_data.size()}')

# # negative guess noisy data
# neg_noisy_u = u_negative_initial_guess[16000:,:,:]
# print(f'neg_noisy_u -- {neg_noisy_u.size()}') # 320000*64*1
# neg_noisy_x = x_negative_initial_guess[16000:,:]
# print(f'neg_noisy_x -- {neg_noisy_x.size()}') # 320000*5



# ###### Plot ######
# num_i = 80
# step = np.linspace(0,num_i-1,num_i)
# step_u = np.linspace(0,num_i-1,num_i)

# # # noisy data reshape
# # plot_pos_noisy_u = pos_noisy_u[0:1600,0,0] # 1600*1*1
# # print(f'plot_pos_noisy_u  -- {plot_pos_noisy_u .size()}')
# # # plot_pos_noisy_u = plot_pos_noisy_u.arange(1600).view(1600,1,1)
# # plot_pos_noisy_u_reshape = plot_pos_noisy_u.view(80, 20)
# # # plot_pos_noisy_u_reshape = plot_pos_noisy_u_reshape.t()

# # plot_pos_noisy_x = pos_noisy_x[0:1600,:] # 1600*5
# # print(f'plot_pos_noisy_x  -- {plot_pos_noisy_x .size()}')

# # # pos position
# # plot_pos_noisy_position = plot_pos_noisy_x[:,0]
# # print(f'first 20 noise position -- {plot_pos_noisy_position[0:20]}')
# # # plot_pos_noisy_position = plot_pos_noisy_position.arange(1600).view(1600, 1)
# # plot_pos_noisy_position_reshape = torch.zeros(20,80)
# # n = 0
# # for i in range(0,80):
# #     plot_pos_noisy_position_reshape[:,i] = plot_pos_noisy_position[n*20:n*20+20]
# #     n += 1
# # # plot_pos_noisy_position_reshape = plot_pos_noisy_position.view(80, 20)
# # # plot_pos_noisy_position_reshape = plot_pos_noisy_position_reshape.t()
# # plot_pos_noisy_position_reshape = plot_pos_noisy_position_reshape.t()
# plot_pos_noisy_position_reshape = plot_pos_noisy_position_reshape.t()

# # # pos velocity
# # plot_pos_noisy_velocity = plot_pos_noisy_x[:,1]
# # # plot_pos_noisy_velocity = plot_pos_noisy_velocity.arange(1600).view(1600, 1)
# # plot_pos_noisy_velocity_reshape = plot_pos_noisy_velocity.view(80, 20)
# # # plot_pos_noisy_velocity_reshape = plot_pos_noisy_velocity_reshape.t()

# # # pos theta
# # plot_pos_noisy_theta = plot_pos_noisy_x[:,2]
# # # plot_pos_noisy_theta = plot_pos_noisy_theta.arange(1600).view(1600, 1)
# # plot_pos_noisy_theta_reshape = plot_pos_noisy_theta.view(80, 20)
# # # plot_pos_noisy_theta_reshape = plot_pos_noisy_theta_reshape.t()

# # # pos theta_dot
# # plot_pos_noisy_theta_dot = plot_pos_noisy_x[:,3]
# # # plot_pos_noisy_theta_dot = plot_pos_noisy_theta_dot.arange(1600).view(1600, 1)
# # plot_pos_noisy_theta_dot_reshape = plot_pos_noisy_theta_dot.view(80, 20)
# # # plot_pos_noisy_theta_dot_reshape = plot_pos_noisy_theta_dot_reshape.t()

# # # pos theta_star
# # plot_pos_noisy_theta_star = plot_pos_noisy_x[:,4]
# # # plot_pos_noisy_theta_star = plot_pos_noisy_theta_star.arange(1600).view(1600, 1)
# # plot_pos_noisy_theta_star_reshape = plot_pos_noisy_theta_star.view(80, 20)
# # # plot_pos_noisy_theta_star_reshape = plot_pos_noisy_theta_star_reshape.t()

# # print(f'noisy_x_along_control_steps -- {plot_pos_noisy_position_reshape[:,0]}')


# plt.figure(figsize=(12, 8))

# plt.subplot(6, 1, 1)
# plt.plot(step, x_noi_data_pos_first_position_reshape[:,:])
# plt.plot(step, pos_orig_x_data[0:80, 0])
# plt.plot(step, x_noi_data_neg_first_position_reshape[:,:])
# plt.plot(step, neg_orig_x_data[0:80, 0])
# # plt.legend(['noise 1','MPC']) 
# plt.ylabel('position (m)')
# plt.grid()

# plt.subplot(6, 1, 2)
# plt.plot(step, x_noi_data_pos_first_velocity_reshape[:,:])
# plt.plot(step, pos_orig_x_data[0:80, 1])
# plt.plot(step, x_noi_data_neg_first_velocity_reshape[:,:])
# plt.plot(step, neg_orig_x_data[0:80, 1])
# plt.ylabel('velocity (m/s)')
# plt.grid()

# plt.subplot(6, 1, 3)
# plt.plot(step, x_noi_data_pos_first_theta_reshape[:,:])
# plt.plot(step, pos_orig_x_data[0:80, 2])
# plt.plot(step, x_noi_data_neg_first_theta_reshape[:,:])
# plt.plot(step, neg_orig_x_data[0:80, 2])
# plt.ylabel('theta (rad)')
# plt.grid()

# plt.subplot(6, 1, 4)
# plt.plot(step, x_noi_data_pos_first_theta_dot_reshape[:,:])
# plt.plot(step, pos_orig_x_data[0:80, 3])
# plt.plot(step, x_noi_data_neg_first_theta_dot_reshape[:,:])
# plt.plot(step, neg_orig_x_data[0:80, 3])
# plt.ylabel('theta_dot (rad/s)')
# plt.grid()

# plt.subplot(6, 1, 5)
# plt.plot(step, x_noi_data_pos_first_theta_star_reshape[:,:])
# plt.plot(step, pos_orig_x_data[0:80, 4])
# plt.plot(step, x_noi_data_neg_first_theta_star_reshape[:,:])
# plt.plot(step, neg_orig_x_data[0:80, 4])
# plt.ylabel('theta star (rad)')
# plt.grid()

# plt.subplot(6, 1, 6)
# plt.plot(step_u, u_noi_data_pos_first_reshape[:,:])
# plt.plot(step_u, pos_orig_u_data[0:80,0,0])
# plt.plot(step_u, u_noi_data_neg_first_reshape[:,:])
# plt.plot(step_u, neg_orig_u_data[0:80,0,0])
# plt.ylabel('Ctl Input (N)')
# plt.xlabel('Control Step')
# plt.grid()
# # plt.show()

# # save figure 
# results_dir = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/mpc_data_collecting'
# figure_name = '2_guess_test.png'
# figure_path = os.path.join(results_dir, figure_name)
# plt.savefig(figure_path)
