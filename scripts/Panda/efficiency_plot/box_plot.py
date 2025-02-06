import numpy as np
import matplotlib.pyplot as plt
import os

figure_saving_path ='/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot'

# Load data from the .npy files
file_path_1 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot/concatenated_solving_data.npy"  # Replace with your first .npy file path
file_path_2 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot/concatenated_nmpc_solving_data.npy"  # Replace with your second .npy file path
# file_path_1 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/concatenated_diffusion_cost.npy"  # Replace with your first .npy file path
# file_path_2 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/model_performance_saving/Panda_test6_performance/concatenated_nmpc_cost.npy"  # Replace with your second .npy file path

data1 = np.load(file_path_1)
data1 = data1.squeeze()
data2 = np.load(file_path_2)
data2 = data2.squeeze()

# Combine the data for comparison
data = [data1, data2]  # Group data into a list for the box plot

# Draw the box plot
plt.figure(figsize=(8, 6))
box = plt.boxplot(data, labels=["Diffusion", "NMPC"], patch_artist=True)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
for b in box['boxes']:
    b.set_facecolor("blue") # green

# Customize the plot
plt.title("Computation Time for Franka Panda Control policy", fontsize=18, fontweight='bold')
plt.xlabel("Methods", fontsize=16, fontweight='bold')
plt.ylabel("Time (seconds)", fontsize=16, fontweight='bold')
plt.grid(axis='y')

# plt.title("Optimization Cost for Franka Panda Motion Task", fontsize=18, fontweight='bold')
# plt.xlabel("Methods", fontsize=16, fontweight='bold')
# plt.ylabel("Cost", fontsize=16, fontweight='bold')
# plt.grid(axis='y')

# Get the median values from the box plot object
medians = [item.get_ydata()[0] for item in box['medians']]

for median_line in box['medians']:
    median_line.set_linewidth(3)

# Add the median values as text annotations
for i, median in enumerate(medians):
    plt.text(i + 1 - 0.1, median, f'{median:.2f}', ha='right', va='center', color='chocolate', fontsize=16, fontweight='bold')
    # if i == 0:  
    #     # plt.text(i + 1 + 0.1, median, '$A_{df}$', ha='left', va='center', color='blue', fontsize=16)
    #     plt.text(i + 1 + 0.1, median, '$DT_{1}$', ha='left', va='center', color='blue', fontsize=16, fontweight='bold')

# Add labels to outliers
for i, flier in enumerate(box['fliers']):
    outliers = flier.get_ydata()
    # if i == 0:  # First dataset (Diffusion)
    #     # Add custom labels to the first two outliers
    #     plt.text(i + 1 + 0.1, outliers[0] + 100, '$DT_{5}$', ha='left', va='center', color='blue', fontsize=16, fontweight='bold')
    #     plt.text(i + 1 + 0.1, outliers[1 ]- 100 , '$DT_{4}$', ha='left', va='center', color='blue', fontsize=16, fontweight='bold')

# Show the plot
# plt.show()
figure_name = 'solving efficiency comparision_new_03' + '.pdf'
figure_path = os.path.join(figure_saving_path, figure_name)
plt.savefig(figure_path)  


# ### compare problem solving time ###

# # Load data from the .npy files
# solving_path_1 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot/concatenated_solving_data.npy"  # Replace with your first .npy file path
# solving_path_2 = "/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/Panda/efficiency_plot/concatenated_nmpc_solving_data.npy"  # Replace with your second .npy file path

# solving_data1 = np.load(solving_path_1)
# solving_data1 = solving_data1.squeeze()
# solving_data2 = np.load(solving_path_2)
# solving_data2 = solving_data2.squeeze()

# # Combine the data for comparison
# solving_data = [solving_data1, solving_data2]  # Group data into a list for the box plot

# # Draw the box plot
# plt.figure(figsize=(8, 6))
# solving_box = plt.boxplot(solving_data, labels=["Diffusion", "NMPC"], patch_artist=True)

# # Customize the plot
# plt.title("Computation Time for Franka Panda Optimization")
# plt.xlabel("Methods")
# plt.ylabel("Time (seconds)")
# plt.grid(axis='y')

# # Get the median values from the box plot object
# medians = [item.get_ydata()[0] for item in solving_box['medians']]

# # Add the median values as text annotations
# for i, median in enumerate(medians):
#     plt.text(i + 1 - 0.1, median, f'{median:.2f}', ha='right', va='center', color='chocolate', fontsize=10)

# # Show the plot
# # plt.show()
# solving_figure_name = 'solving efficiency' + '.pdf'
# solving_figure_path = os.path.join(figure_saving_path, solving_figure_name)
# plt.savefig(solving_figure_path)  
