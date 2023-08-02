# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:27:44 2023

@author: ULTRASIP_1
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a list to store the figures for each frame
fig_list = []

# Sample data
distance_km = np.array([0, 0.24655, 0.50548, 1.0288,1.3457,1.5796,2.0224,2.5133,3.0068,3.2842,3.778,4.77308,5.2714])
scattering_angles_deg = np.array([38.54, 52.33, 73.17, 91.83, 101.71])

# Generate sample DoLP values
doLP355 = np.array([
    [0.101, 0.1, 0.101, 0.101,0.101,0.069, 0.099, 0.075,	0.084,	0.089,	0.113,	0.074,	0.088],
    [0.247, 0.245, 0.246, 0.246,0.247,	0.188,	0.242,	0.198,	0.218,	0.226,	0.265,	0.199,	0.223],
    [0.451, 0.449, 0.45, 0.449,0.451, 0.361,	0.443,	0.378,	0.405,	0.419,	0.483,	0.378,0.412],
    [0.516, 0.514, 0.515, 0.515,0.517,	0.424,	0.509,	0.445,	0.466,	0.484,	0.554,0.442,0.476],
    [0.474, 0.472, 0.473, 0.473,0.474,	0.395,0.468,	0.415,	0.429,	0.446,	0.509,	0.41,0.44]
])

doLP555 = np.array([
    [0.116,	0.117,	0.117,	0.115,	0.114,	0.058,	0.124,	0.078,	0.114,	0.104	,0.09,	0.33,	0.317],
    [0.244,	0.248,	0.248,	0.243,	0.24,	0.137,	0.26,	0.18,	0.24,	0.224,	0.208,	0.22,	0.214],
    [0.477,	0.485,	0.484,	0.476,	0.469,	0.289,	0.504,	0.377,	0.47,	0.444,	0.433,	0.18,	0.188],
    [0.587,	0.595	,0.595	,0.586	,0.579,	0.371,0.615,	0.479,	0.58,	0.551,	0.542,	0.314,	0.343],
    [0.558,	0.566,	0.565,	0.559,	0.553,	0.359,	0.583,	0.461,	0.554,	0.526,	0.517,	0.444,0.486]
])

colors = np.array(['red','blue','green','purple','orange','pink','gray'])


# Assuming you have 'distance_km' and 'doLP555' from the previous code
# Replace 1 with the row index of doLP555 that you want to plot
# for index in range(13):

#     # Create another figure for the second plot
#     plt.figure()
#     # Plot the scatter points for the specified row
#     plt.plot(scattering_angles_deg, doLP555[:,index], color=colors[2], linestyle="solid",label='555nm')
#     plt.plot(scattering_angles_deg, doLP355[:,index], color=colors[1], linestyle="solid",label='355nm')

#     # Set labels and title for the second plot
#     plt.xlabel('Scattering Angle')
#     plt.ylabel('DoLP Values')
#     plt.title('Distance {} [km]'.format(np.round(distance_km[index],1)))
#     plt.legend(ncol=2)
#     # Show the second plot
#     plt.show()

# Create a figure and axis for the plots
# fig, ax = plt.subplots()

# # Set the y-axis limits to be the same for all plots
# y_min = min(np.min(doLP555), np.min(doLP355))
# y_max = max(np.max(doLP555), np.max(doLP355))
# ax.set_ylim(y_min, y_max)

# # Create a list to store the line objects for each plot
# lines = []

# # Assuming you have 'distance_km' and 'doLP555' from the previous code
# # Replace 1 with the row index of doLP555 that you want to plot
# for index in range(13):
#     # Plot the scatter points for the specified row
#     line555, = ax.plot(scattering_angles_deg, doLP555[:, index], color=colors[2], linestyle="solid", label='555nm')
#     line355, = ax.plot(scattering_angles_deg, doLP355[:, index], color=colors[1], linestyle="solid", label='355nm')

#     # Set labels and title for the plot
#     ax.set_xlabel('Scattering Angle')
#     ax.set_ylabel('DoLP Values')
#     ax.set_title('Distance {} [km]'.format(np.round(distance_km[index], 1)))

#     # Add legend to the plot
#     ax.legend(loc='upper left', ncol=2)

#     # Store the line objects in the list
#     lines.append([line555, line355])

# # Function to update the plot for each frame
# def update(frame):
#     ax.clear()
#     ax.set_xlabel('Scattering Angle')
#     ax.set_ylabel('DoLP Values')
#     ax.set_title('Distance {} [km]'.format(np.round(distance_km[frame], 1)))

#     # Set the y-axis limits to be the same for all plots
#     ax.set_ylim(y_min, y_max)

#     # Plot the scatter points for the specified row
#     line555, = ax.plot(scattering_angles_deg, doLP555[:, frame], color=colors[2], linestyle="solid", label='555nm')
#     line355, = ax.plot(scattering_angles_deg, doLP355[:, frame], color=colors[1], linestyle="solid", label='355nm')

#     # Add legend to the plot
#     ax.legend(loc='upper left', ncol=2)

#     return line555, line355

# # Create the animation using FuncAnimation
# animation = FuncAnimation(fig, update, frames=len(lines), blit=False, repeat=False)

# # Save the animation as a video (you can change the file format if needed, e.g., .mp4, .avi, etc.)
# animation.save('doLP_plots_animation.mp4', writer='ffmpeg', fps=1)

# # Show the last plot (optional)
# plt.show()






# Create a meshgrid for distance and scattering angles
distance_mesh, scattering_angles_mesh = np.meshgrid(distance_km, scattering_angles_deg)

# Flatten the DoLP values to match the shape of the meshgrid
doLP355_flattened = doLP355.flatten()
doLP555_flattened = doLP555.flatten()


# # Create a meshgrid for distance and scattering angles
# distance_mesh, scattering_angles_mesh = np.meshgrid(distance_km, scattering_angles_deg)

# Set the figure size (width, height) in inches
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points as scatter
ax.scatter(distance_mesh.flatten(), scattering_angles_mesh.flatten(), doLP355.flatten(), c='blue')

# Connect dots with lines
for i in range(5):
    ax.plot(distance_km, np.full_like(distance_km, scattering_angles_deg[i]), doLP355[i], color=colors[i])

# Plot the data points as scatter
ax.scatter(distance_mesh.flatten(), scattering_angles_mesh.flatten(), doLP555.flatten(), c='blue')

# Connect dots with lines
for i in range(5):
    ax.plot(distance_km, np.full_like(distance_km, scattering_angles_deg[i]), doLP555[i], color=colors[i],linestyle="dashed")

# Set labels and title
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Scattering Angles (degrees)')
ax.set_zlabel('DoLP Values')

# Shift perspective by adjusting elevation and azimuth angles
ax.view_init(elev=20, azim=-50)

# Show the plot
plt.show()






