# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:14:58 2023

@author: ULTRASIP_1
"""

#Script to choose ROI

# Import packages

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches



def image_crop(a):
        #np.clip(a, 0, None, out=a)
        a[a == -999] = np.nan
        mid_row = a.shape[0] // 2
        mid_col = a.shape[1] // 2
        start_row = mid_row - 524
        end_row = mid_row + 524
        start_col = mid_col - 524
        end_col = mid_col + 524
        
        a = a[start_row:end_row, start_col:end_col]
        return a


def calculate_std(image):
# Define the size of the regions we'll calculate the standard deviation for
    region_size = 5

    # Calculate the standard deviation over the regions
    std_dev = np.zeros_like(image)
    for i in range(region_size//2, image.shape[0] - region_size//2):
        for j in range(region_size//2, image.shape[1] - region_size//2):
            std_dev[i,j] = np.std(image[i-region_size//2:i+region_size//2+1, j-region_size//2:j+region_size//2+1])

    return std_dev

def calculate_median(image):
# Define the size of the regions we'll calculate the standard deviation for
    region_size = 5

    # Calculate the standard deviation over the regions
    median_img = np.zeros_like(image)
    for i in range(region_size//2, image.shape[0] - region_size//2):
        for j in range(region_size//2, image.shape[1] - region_size//2):
            median_img[i,j] = np.median(image[i-region_size//2:i+region_size//2+1, j-region_size//2:j+region_size//2+1])

    return median_img
# Start the main code

def  choose_roi(image): 
            std_dev = calculate_std(image)
            med_img = calculate_median(image)
    # Plot the original image and the standard deviation image side by side
            fig, ax = plt.subplots(1,2,  figsize=(16, 8))
            ax[0].imshow(image , cmap = 'gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            im = ax[1].imshow(std_dev, cmap = 'jet')
            ax[1].set_title('Standard Deviation')
            ax[1].grid(True)
            cbar = fig.colorbar(im, ax = ax[1], fraction = 0.046, pad=0.04)
            
            plt.show()

        # Prompt the user to choose a region
            x = int(input('Enter x-coordinate of region: '))
            y = int(input('Enter y-coordinate of region: '))

          
            # Create a new figure with 1 row and 2 columns
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Plot the original image with the selected region of interest highlighted
            axs[0].imshow(image, cmap='gray')
            axs[0].add_patch(patches.Rectangle((x, y), 5, 5, linewidth=5, edgecolor='w', facecolor='none'))
            axs[0].set_title('Selected Region of Interest')

            # Plot the standard deviation image with the selected region of interest highlighted
            im = axs[1].imshow(std_dev, cmap='jet')
            axs[1].add_patch(patches.Rectangle((x, y),5,5,linewidth=5, edgecolor='w', facecolor='none'))
            axs[1].set_title('Standard Deviation with Selected Region of Interest')
            cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        # Show the plot
            plt.show()
            
            print('Standard deviation of region:', std_dev[x,y])
            region_data = med_img[x, y]
            print('Median image value is:',region_data)
            
            return x,y

def main():  # Main code


# Set the paths
# NOTE: basepath is the location of the AirMSPI HDF data files
#       figpath is where the output should be stored

    #basepath = "C:/Users/ULTRASIP_1/Documents/Bakersfield707_DataCopy/"
    basepath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    #figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Retrieval_Files/Plots"
    figpath = 'C:/Users/ULTRASIP_1/Documents/Clarissa/paperfigs'

    
# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 1
    
# Set the index of the group of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 0
    
    os.chdir(basepath)

# Get the list of files in the directory
# NOTE: Python returns the files in a strange order, so they will need to be sorted by time

    search_str = 'AirMSPI*467A*.hdf'
    dum_list = glob.glob(search_str)
    raw_list = np.array(dum_list)  # Convert to a numpy array
    
# Get the number of files
    
    num_files = len(raw_list)
    
# Check the number of files against the index

    print("AirMSPI Files Found: ",num_files)
    
    num_need = num_step*step_ind+num_step
    
    if(num_need > num_files):
        print("***ERROR***")
        print("Not Enough Files for Group Index")
        print(num_need)
        print(num_files)
        print("Check index of the group at Line 43")
        print("***ERROR***")
        print(error)

# Loop through files and sort by time (HHMMSS)

    time_raw = np.zeros((num_files),dtype=int)

    for loop in range(num_files):
    
# Select appropriate file

        this_file = raw_list[loop]

# Parse the filename to get information
    
        words = this_file.split('_')
        temp = words[5]
        hold = temp.split('Z')
        time_hhmmss = int(hold[0])
        time_raw[loop] = time_hhmmss

# Sort the files

    sorted = np.argsort(time_raw)
    mspi_list = raw_list[sorted]
    time_list = time_raw[sorted]    
    
    
### Loop through the files for one set

    for loop in range(num_step):
    
        this_ind = loop+num_step*step_ind
                
# Get the filename

        inputName = basepath+'/'+mspi_list[this_ind]
        
# Tell user location in process

        print("Reading: "+inputName)

#Get time stamp and andle 
        words = mspi_list[this_ind].split('_')
        timeoffile_hhmmss = words[5]
        angleoffile = words[7]
        
# Open the HDF-5 file

        f = h5py.File(inputName,'r')
        


# Get the intensity datasets

        I_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]  
        I_355 = image_crop(I_355)

    
        I_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:]
        I_380 = image_crop(I_380)

        I_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:]
        I_445 = image_crop(I_445)

        I_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]
        I_470 = image_crop(I_470)

        I_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:]
        I_555 = image_crop(I_555)


        I_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:]
        I_660 = image_crop(I_660)

        I_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:]
        I_865 = image_crop(I_865)



    
    
# Close the file

        f.close()

        img355 = np.flipud(I_355)
        img380= np.flipud(I_380)
        img445= np.flipud(I_445)

        img470= np.flipud(I_470)
        img555= np.flipud(I_555)
        img660= np.flipud(I_660)


        img865 = np.flipud(I_660)
        

        roi_x, roi_y = choose_roi(img660)
    return roi_x, roi_y
        

### END MAIN FUNCTION
if __name__ == '__main__':
        x,y = main() 
        
        print(x,y)

        
        


      