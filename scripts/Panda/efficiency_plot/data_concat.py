import torch
import numpy as np
import os

FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot' 

# npy load
npy_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/cost_collecting_sample_1/cost_.npy'
npy_data = np.load(npy_path)
print(f'time -- {npy_data}')

npy_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/single_time_diffusion_.npy'
data = np.load(npy_path)
print(f'delta time -- {data.shape}')

single_file_1 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/single_time_diffusion_.npy'
single_1 = np.load(single_file_1)
single_file_2 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_17test_pdf/single_time_diffusion_.npy'
single_2 = np.load(single_file_2)
single_file_3 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_19test_pdf/single_time_diffusion_.npy'
single_3 = np.load(single_file_3)
single_file_4 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_21test_pdf/single_time_diffusion_.npy'
single_4 = np.load(single_file_4)
single_file_5 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_23test_pdf/single_time_diffusion_.npy'
single_5 = np.load(single_file_5)


solving_file_1 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/solving_time_diffusion_.npy'
solving_1 = np.load(solving_file_1)
solving_1 = solving_1.reshape(1,1)
solving_file_2 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_17test_pdf/solving_time_diffusion_.npy'
solving_2 = np.load(solving_file_2)
solving_2 = solving_2.reshape(1,1)
solving_file_3 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_19test_pdf/solving_time_diffusion_.npy'
solving_3 = np.load(solving_file_3)
solving_3 = solving_3.reshape(1,1)
solving_file_4 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_21test_pdf/solving_time_diffusion_.npy'
solving_4 = np.load(solving_file_4)
solving_4 = solving_4.reshape(1,1)
solving_file_5 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_23test_pdf/solving_time_diffusion_.npy'
solving_5 = np.load(solving_file_5)
solving_5 = solving_5.reshape(1,1)

concatenated_single_data = np.concatenate((single_1, single_2, single_3, single_4, single_5), axis=0)
single_file_path = os.path.join(FOLDER_PATH, "concatenated_single_data.npy")
print(f'single size -- {concatenated_single_data.shape}')
np.save(single_file_path, concatenated_single_data)


concatenated_solving_data = np.concatenate((solving_1, solving_2, solving_3, solving_4, solving_5), axis=0)
solving_file_path = os.path.join(FOLDER_PATH, "concatenated_solving_data.npy")
print(f'solving size -- {concatenated_solving_data.shape}')
np.save(solving_file_path, concatenated_solving_data )

# nmpc data concate
nmpc_single_file_1 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/time_mpc_idx-101.npy'
nmpc_single_1 = np.load(nmpc_single_file_1 )
print(f'delta time -- {nmpc_single_1}')
nmpc_single_file_2 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/time_mpc_idx-102.npy'
nmpc_single_2 = np.load(nmpc_single_file_2 )
nmpc_single_file_3 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/time_mpc_idx-103.npy'
nmpc_single_3 = np.load(nmpc_single_file_3 )

nmpc_solving_file_1 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/solving_time_mpc_idx-101.npy'
nmpc_solving_1 = np.load(nmpc_solving_file_1 )
nmpc_solving_1 = nmpc_solving_1.reshape(1,1)
print(f'solving time -- {nmpc_solving_1}')
nmpc_solving_file_2 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/solving_time_mpc_idx-102.npy'
nmpc_solving_2 = np.load(nmpc_solving_file_2 )
nmpc_solving_2 = nmpc_solving_2.reshape(1,1)
nmpc_solving_file_3 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/nmpc_time_data/solving_time_mpc_idx-103.npy'
nmpc_solving_3 = np.load(nmpc_solving_file_3 )
nmpc_solving_3 = nmpc_solving_3.reshape(1,1)

concatenated_nmpc_single_data = np.concatenate((nmpc_single_1, nmpc_single_2, nmpc_single_3), axis=0)
single_file_path = os.path.join(FOLDER_PATH, "concatenated_nmpc_single_data.npy")
print(f'single size -- {concatenated_nmpc_single_data.shape}')
np.save(single_file_path, concatenated_nmpc_single_data )


concatenated_nmpc_solving_data = np.concatenate((nmpc_solving_1, nmpc_solving_2, nmpc_solving_3), axis=0)
solving_file_path = os.path.join(FOLDER_PATH, "concatenated_nmpc_solving_data.npy")
print(f'solving size -- {concatenated_nmpc_solving_data.shape}')
np.save(solving_file_path, concatenated_nmpc_solving_data )