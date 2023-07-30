# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:13:26 2023

@author: ULTRASIP_1
"""

#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.ticker as ticker


def format_directory():
    
    directory_path = input("Enter the directory path: ")
    # Remove trailing slashes
    directory_path = directory_path.rstrip(os.path.sep)

    # Normalize path separators
    directory_path = os.path.normpath(directory_path)

    # Replace backslashes with forward slashes
    directory_path = directory_path.replace('\\', '/')

    # Format the directory path as a raw string literal
    formatted_directory_path = r"{}".format(directory_path)

    return formatted_directory_path


def load_hdf_files(folder_path, group_size,idx):
    # Get a list of HDF file paths in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*.hdf'))

    # Sort file paths based on date and time in the file name
    sorted_file_paths = sorted(file_paths, key=lambda x: (os.path.basename(x).split('_')[5][0:8], os.path.basename(x).split('_')[5][8:14]))

    # Load files in groups of the specified size
    groups = [sorted_file_paths[i:i+group_size] for i in range(0, len(sorted_file_paths), group_size)]

    # Choose one group to use
    selected_group = groups[idx]  # Change the index to select a different group

    return selected_group

def image_format(image):
    
    image[np.where(image == -999)] = np.nan
    #new_image = image[1100:2800,500:2200]
    new_image = image[1900:2400,1100:1750]
    
    return new_image

#Variable Defns
num_step = 5
step_ind = 0
esd = 0.0
num_int = 8 
num_pol = 3
num_meas = 13

datapath = format_directory()



data_files = load_hdf_files(datapath, num_step,step_ind)

nadir = [string for string in data_files if '000N' in string][0]

f = h5py.File(nadir,'r')

i_445 = image_format(f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:])
i_555 = image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:])
i_660 = image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:])
latitude = image_format(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/'][:])
longitude = image_format(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][:])


i_rgb = (i_445+i_555+i_660)

f.close()

# plt.figure(figsize=(10, 10))
# plt.imshow(i_rgb)
# plt.title('I_RGB at Nadir')

# plt.show()

# Assuming the data needs to be rescaled to the range [0, 255]
i_445_rescaled = (i_445 - np.nanmin(i_445)) * 255.0 / (np.nanmax(i_445) - np.nanmin(i_445))
i_555_rescaled = (i_555 - np.nanmin(i_555)) * 255.0 / (np.nanmax(i_555) - np.nanmin(i_555))
i_660_rescaled = (i_660 - np.nanmin(i_660)) * 255.0 / (np.nanmax(i_660) - np.nanmin(i_660))

# # Combining the rescaled channels to create the RGB image
# i_rgb = np.stack([i_660_rescaled, i_555_rescaled, i_445_rescaled], axis=-1)

# plt.figure(figsize=(10, 10))
# plt.imshow(i_rgb.astype(np.uint8))  # Convert to uint8 to ensure correct data type for displaying the image
# plt.title('I_RGB at Nadir')
# plt.axis('on')
# plt.show()

# Combining the rescaled channels to create the RGB image
#i_rgb = np.stack([i_660_rescaled, i_555_rescaled, i_445_rescaled], axis=-1)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ... (existing code) ...

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ... (existing code) ...

# Combining the rescaled channels to create the RGB image
i_rgb = np.stack([i_660_rescaled, i_555_rescaled, i_445_rescaled], axis=-1)

# Normalize the RGB data to the range [0, 1]
i_rgb = i_rgb / 255.0

# Define the geographical extent of the plot
min_lon, max_lon = np.min(longitude), np.max(longitude)
min_lat, max_lat = np.min(latitude), np.max(latitude)

# Create a Cartopy PlateCarree projection (unprojected geographic coordinates)
projection = ccrs.PlateCarree()

# Create the figure and axis
plt.figure(figsize=(6, 4))
ax = plt.axes(projection=projection)

# Plot the RGB image using imshow and latitude/longitude extents
plt.imshow(i_rgb, origin='upper', extent=[min_lon, max_lon, min_lat, max_lat],
           transform=projection)

# Add coastlines and gridlines
ax.coastlines(resolution='10m', color='black', linewidth=0.5)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Set the title and labels with increased font size
#plt.title('RGB Image with Georeferenced Plot', fontsize=18)
# plt.xlabel('Longitude', fontsize=50)
# plt.ylabel('Latitude', fontsize=50)

# Latitude and Longitude pairs (Replace these with your own 12 coordinates)
lat_lon_pairs = [
    (47.9392, -118.4957),
    (47.9392, -118.4924),
    (47.9402,-118.4891),
    (47.9416,-118.4824),
    (47.9439,-118.4791),
    (47.9440,-118.4758),
    (47.9422,-118.4690),
    (47.9419,-118.4623),
    (47.9415,-118.4556),
    (47.9353,-118.4521),
    (47.9443,-118.4457),
    (47.9446,-118.4256)
]

# Extract latitudes and longitudes from the pairs
latitudes, longitudes = zip(*lat_lon_pairs)

# Plot the dots on the map
ax.scatter(longitudes, latitudes, color='cyan', s=60, transform=projection)


# Show the plot
plt.show()


