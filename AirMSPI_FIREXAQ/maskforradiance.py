# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:43:53 2023

@author: Clarissa
"""

def image_crop(a):
    a = a[~(a == -999).all(axis=1)]
    a = a[:, ~(a == -999).all(axis=0)]
    a[np.where(a == -999)] = np.nan

    mid_row = a.shape[0] // 2
    mid_col = a.shape[1] // 2

    # Define the desired crop size (2096x2096)
    crop_size = 2096

    # Calculate start and end row indices for cropping
    start_row = max(mid_row - (crop_size // 2), 0)
    end_row = start_row + crop_size

    # Calculate start and end column indices for cropping
    start_col = max(mid_col - (crop_size // 2), 0)
    end_col = start_col + crop_size

    # Crop the image to the desired size (2096x2096)
    a = a[start_row:end_row, start_col:end_col]

    # Check if the cropped array is smaller than 2096x2096 and pad if necessary
    if a.shape[0] < crop_size:
        padding_rows = crop_size - a.shape[0]
        a = np.pad(a, ((0, padding_rows), (0, 0)), mode='constant', constant_values=np.nan)
    if a.shape[1] < crop_size:
        padding_cols = crop_size - a.shape[1]
        a = np.pad(a, ((0, 0), (0, padding_cols)), mode='constant', constant_values=np.nan)

    return a

def update(frame_number):
    plt.clf()
    plt.imshow(I_med_list[frame_number], cmap='copper')
    plt.title(title_list[frame_number])
    colorbar = plt.colorbar()
    plt.grid(True)

#mask making
#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches
import matplotlib.animation as animation

#_______________Start the main code____________#
def main():  # Main code

#___________________Section 1: Data Products______________________#

#Load in Data
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/

# Set paths to AirMSPI data and where the output SDATA file should be saved 
# NOTE: datapath is the location of the AirMSPI HDF data files
#       outpath is where the output should be stored
#Work Computer
    datapath = "C:/Users/Clarissa/Documents/AirMSPI/Inchelium"
    outpath = "C:/Users/Clarissa/Documents/Github/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/July1723"

# # Load in the set of measurement sequences
# Set the length of one measurement sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    
# Calculate the middle of the sequence

    mid_step = int(num_step/2)  
    
# Set the index of the sequence of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 0
    
# Set the number of wavelengths for radiometric and polarization separately
#num_int = total number of radiometric channels
#num_pol = total number of polarimetric channels

    num_int = 8 
    num_pol = 3
    
# Create arrays to store data
# NOTE: Pay attention to the number of wavelengths
    num_meas = 13
    
# Change directory to the datapath
    os.chdir(datapath)

# Get the list of files in the directory
# NOTE: Python returns the files in a strange order, so they will need to be sorted by time
    #Search for files with the correct names
    search_str = '*TERRAIN*.hdf'
    dum_list = glob.glob(search_str)
    raw_list = np.array(dum_list)  # Convert to a numpy array
    
# Get the number of files    
    num_files = len(raw_list)
    
# Check the number of files against the index to only read one measurement sequence
    print("AirMSPI Files Found: ",num_files)
    

    I = np.zeros((2096,2096))
    I_med_list = []  # List to store the I_med images
    title_list = []  # List to store the titles
    
#Start the for loop
    for loop in range(num_files):
    
                
# Get the filename

        inputName = datapath+'/'+raw_list[loop]
        
# Tell user location in process

        print("Reading: "+inputName)
        
# Open the HDF-5 file

        f = h5py.File(inputName,'r')
        
#_________________________Read the data_________________________#
# Set the datasets and read (355 nm)
# Radiometric Channel

        print("355nm")
        I_355 = image_crop(f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]) 
        I_380 = image_crop(f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:])
        I_445 = image_crop(f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:])
        I_470 = image_crop(f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:])
        I_555 = image_crop(f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:])
        I_660 = image_crop(f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:])
        I_865 = image_crop(f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:])
        
        I_med = I_355 + I_380 + I_445 + I_470 + I_555 + I_660 + I_865
        I = I + I_med
        
        I_med_list.append(I_med)  # Store the I_med image in the list
        title_list.append(f'Mask: {inputName[79:86]}, {inputName[100:104]}')
        
        plt.figure()
        plt.imshow(I_med,cmap='copper')
        plt.title('Mask:' +inputName[79:86]+' ,'+inputName[100:104])
        plt.grid(True)
        colorbar = plt.colorbar()
        colorbar.set_ticks(np.arange(0.25, 0.85, 0.05))

        f.close()

    return I_med_list,title_list,I
    
    
if __name__ == "__main__":
    I_med_list, title_list, I = main()
    
    plt.figure()
    plt.imshow(I,cmap='copper')
    plt.title('Final Mask')
    colorbar = plt.colorbar()
    plt.grid(True)
    # Add colorbar


    
    I_med_list.append(I)  # Store the I_med image in the list
    title_list.append('Final Mask')

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=len(I_med_list), interval=1000, repeat=False)

    # Save the animation as a video
    ani.save('I_med_video.mp4', writer='ffmpeg', dpi=300)
    plt.show()