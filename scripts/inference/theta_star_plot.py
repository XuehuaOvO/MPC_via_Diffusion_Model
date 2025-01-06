import numpy as np
import matplotlib.pyplot as plt

theta_value = np.linspace(0,2*np.pi,100)
theta_star = np.zeros(len(theta_value))
for i in range(0, len(theta_value)):
    theta_i = theta_value[i]
    theta_star[i] = np.pi - (theta_i - np.pi)**2/np.pi

point_x = np.pi
point_y = np.pi  # Adjust based on the actual value on the line

plt.figure()
plt.plot(theta_value, theta_star)
plt.scatter(point_x, point_y, color='red', label=f'Point ({point_x:.2f}, {point_y:.2f})')  # Mark the point
plt.axhline(point_y, color='gray', linestyle='--', linewidth=0.5)  # Optional: horizontal line for reference
plt.axvline(point_x, color='gray', linestyle='--', linewidth=0.5)  # Optional: vertical line for reference
plt.scatter(0, 0, color='green', label=f'Point ({0:.2f}, {0:.2f})')  # Mark the point
plt.scatter(2*np.pi, 0, color='green', label=f'Point ({2*np.pi:.2f}, {0:.2f})')  # Mark the point
plt.xlabel("theta")
plt.ylabel("theta star")
plt.legend()
plt.show()