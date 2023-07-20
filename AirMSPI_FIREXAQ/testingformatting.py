# -*- coding: utf-8 -*-

#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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

#Variable Defns

num_step = 9
step_ind = 0
datapath = format_directory()
outpath = format_directory()

data_files = load_hdf_files(datapath, num_step,step_ind)

nadir = [string for string in data_files if '000N' in string][0]

f = h5py.File(nadir,'r')

I_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][1100:2800,500:2200]

I_355[np.where(I_355 == -999)] = np.nan

plt.figure(figsize=(10, 10))
plt.imshow(I_355 , cmap = 'jet')
plt.title('I_355 at Nadir')
plt.grid(True)
plt.colorbar()

# Set the number of gridlines for both the x-axis and y-axis
num_gridlines_x = 60  # Adjust this number as needed
num_gridlines_y = 60  # Adjust this number as needed

# Adjust the number of gridlines using locator_params
plt.gca().xaxis.get_major_locator().set_params(nbins=num_gridlines_x)
plt.gca().yaxis.get_major_locator().set_params(nbins=num_gridlines_y)

plt.gca().tick_params(axis='x', labelrotation=45)

plt.show()

# Get user input for the (x, y) coordinates
try:
    x = int(input("Enter the x-coordinate: "))
    y = int(input("Enter the y-coordinate: "))
except ValueError:
    print("Invalid input. Please enter integer values for coordinates.")
    exit()

# Now, plot the image with the black square marker on top
plt.figure(figsize=(10, 10))
plt.imshow(I_355, cmap='jet')
plt.title('I_355 at Nadir')
plt.grid(True)
plt.colorbar()

# Plot a marker at the chosen (x, y) coordinate on top of the I_355 image
plt.scatter(x, y, c='black', marker='s', facecolors='none', edgecolors='black', s=100)

plt.show()

loop = len(data_files)
for idx in range(loop):
    
    f = h5py.File(data_files[idx],'r')
        
    I_355= f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]

    if np.any(np.isnan(I_355)):
        print("NaN values")
        continue
    else:     
        #get metadata
        words = nadir.split('_')
        date = words[4][0:4] +'-'+words[4][4:6]+'-'+words[4][6:8]
        time = words[5][0:2]+':'+words[5][2:4]+':'+words[5][4:8]
        elev = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/'][1100:2800,500:2200][x,y]
        lat = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/'][1100:2800,500:2200][x,y]
        lon = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][1100:2800,500:2200][x,y]
    
        scat_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/Scattering_angle/'][1100:2800,500:2200][x,y]
        vaz_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_azimuth/'][1100:2800,500:2200][x,y]     
        vza_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_zenith/'][1100:2800,500:2200][x,y]
        
        I_380= f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]     
        scat_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/Scattering_angle/'][1100:2800,500:2200][x,y]   
        vaz_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_azimuth/'][1100:2800,500:2200][x,y]   
        vza_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_zenith/'][1100:2800,500:2200][x,y]   
        
        I_445= f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]
        scat_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/Scattering_angle/'][1100:2800,500:2200][x,y] 
        vaz_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_azimuth/'][1100:2800,500:2200][x,y]
        vza_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_zenith/'][1100:2800,500:2200][x,y]
        
        I_470= f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]
        
        I_555= f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]
        
        I_660= f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]
        
        I_865= f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][1100:2800,500:2200][x,y]

        print(idx,I_355,I_380,I_470,I_555,I_660,I_865)
                        





