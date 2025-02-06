import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# npy load
# npy_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/solving_time_diffusion_.npy'
# npy_data = np.load(npy_path)
# print(f'time -- {npy_data}')
TARGET_POS = np.array([0.3, 0.3, 0.5])
FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/Diffusion_panda_3D_plot'

npy_path = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/x_trag.npy'
data = np.load(npy_path)
print(f'x size -- {data.shape}')

file_x_0 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_0test_pdf/x_trag.npy'
x_0 = np.load(file_x_0 )
x_0 = x_0.reshape(1,200,3)

file_x_19 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_19test_pdf/x_trag.npy'
x_19 = np.load(file_x_19)
x_19 = x_19.reshape(1,200,3)

file_x_23 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_23test_pdf/x_trag.npy'
x_23 = np.load(file_x_23 )
x_23 = x_23.reshape(1,200,3)

file_x_17 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/qPOS_17test_pdf/x_trag.npy'
x_17 = np.load(file_x_17 )
x_17 = x_17.reshape(1,200,3)

file_x_cost4 = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/cost_collecting_sample_4/x_trag.npy'
x_cost4 = np.load(file_x_cost4)
x_cost4 = x_cost4.reshape(1,200,3)

x_concate = np.concatenate((x_0, x_19, x_23, x_17, x_cost4), axis=0)

# plt 3d figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for n in range(0, 5):
        x_states = x_concate[n,:,:]
        ax.plot(x_states[:,0], x_states[:,1], x_states[:,2],label=f"Diffusion Traj {n+1}", linewidth=0.8)
        point = [x_states[-1,0], x_states[-1,1], x_states[-1,2]]
        ax.scatter(point[0], point[1], point[2], color='brown', s=7)
        start_point = [x_states[0,0], x_states[0,1], x_states[0,2]]
        ax.scatter(start_point[0], start_point[1], start_point[2], color='blue', s=10)

ax.scatter(start_point[0], start_point[1], start_point[2], color='blue', s=10, label=f"Initial Position")
ax.scatter(point[0], point[1], point[2], color='brown', s=10, label=f"Final Position")  

target = TARGET_POS
ax.scatter(target[0], target[1], target[2], color='Black', s=40, label=f"Target Position")

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-0.3, 0.3])  # x-axis range
ax.set_ylim([-0.3, 0.3])  # y-axis range
ax.set_zlim([-0.3, 0.3])   # z-axis range
ax.legend()

# initial view
ax.view_init(elev=60, azim=30)

figure_name = 'diffusion_panda_3d_new' + '.pdf'
figure_path = os.path.join(FOLDER_PATH, figure_name)
plt.savefig(figure_path)