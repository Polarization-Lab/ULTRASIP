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

# Start the main code

def main():  # Main code

# Set the overall timer

    all_start_time = time.time()

# Set the paths
# NOTE: basepath is the location of the AirMSPI HDF data files
#       figpath is where the output should be stored

    basepath = "C:/Users/ULTRASIP_1/Documents/Bakersfield707_DataCopy/"
    #basepath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    #figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Retrieval_Files/Plots"
    figpath = 'C:/Users/ULTRASIP_1/Documents/Clarissa/paperfigs'

    
# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    
# Set the index of the group of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 0
    
# Set some bounds for the image (USER INPUT)

    # min_x = 1900
    # max_x = 2200
    # min_y = 1900
    # max_y = 2200
    
    # #Bakersfield
    min_x = 1200
    max_x = 1900
    min_y = 1200
    max_y = 1900
# Set some bounds for the sample box (USER INPUT)
# Note: These coordinates are RELATIVE to the overall bounding box

    # box_x1 = 120
    # box_x2 = 125
    # box_y1 = 105
    # box_y2 = 110
    # box_x1 = 45
    # box_x2 = 50
    # box_y1 = 220
    # box_y2 = 225
    # box_x1 = 135
    # box_x2 = 140
    # box_y1 = 140
    # box_y2 = 145


    #Bakserfield
    box_x1 = 485
    box_x2 = 490
    box_y1 = 485
    box_y2 = 490

### Read the AirMSPI data

# Change directory to the basepath

    os.chdir(basepath)

# Get the list of files in the directory
# NOTE: Python returns the files in a strange order, so they will need to be sorted by time

    search_str = 'AirMSPI*.hdf'
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

        dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/']
        I_355 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/']
        I_380 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/']
        I_445 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/']
        I_470 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/']
        I_555 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/']
        I_660 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/']
        I_865 = dset[:][min_y:max_y,min_x:max_x]
    
        dset = f['/HDFEOS/GRIDS/935nm_band/Data Fields/I/']
        I_935 = dset[:][min_y:max_y,min_x:max_x]
    
# Close the file

        f.close()

# Print the time
                
        img = np.flipud(I_355)

            
        #I_355_rolled = np.lib.stride_tricks.sliding_window_view(img, 5, axis = 0)
        #I_355_rolled = np.lib.stride_tricks.sliding_window_view(img, 5, axis = 1)
        
        #rolled_std = np.std(I_355_rolled, axis=2)

        
    return img
        

### END MAIN FUNCTION
if __name__ == '__main__':
        img = main() 
        
        imgr =  np.lib.stride_tricks.sliding_window_view(img, 5, axis = 0)
        imroll = np.lib.stride_tricks.sliding_window_view(imgr, 5, axis = 1)
        
        std = np.std(imroll,axis=0)


        std_min = np.amin(std)
        std_idx = np.where(std==std_min)
        
       # i1 = Ir[std_idx[0][:],std_idx[1][:],std_idx[2][:],std_idx[2][:]]
       
        box_x1 = (std_idx[0][2])+5
        box_x2 =  (std_idx[0][2])-5
        box_y1 = (std_idx[0][2])+5
        box_y2 = (std_idx[0][2])-5
        
        good = (img > 0.0)
        img_good = img[good]
        
        if(len(img_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print(error)

        img_min = np.amin(img[good])
        img_max = np.amax(img[good])
        img_med = np.median(img[good])
        img_mean = np.mean(img[good])
    
        print("IMAGE RANGE")    
        print(img_min)
        print(img_max)
        print(img_mean)
        print(img_med)
        
        stdcheck = np.std(img[box_x2:box_x1,box_y2:box_y1])
        plot_min = 0.7*img_mean
        plot_max = 1.5*img_mean

        hold = np.shape(img)
        xd = hold[1]
        yd = hold[0]
        xdim = np.arange(hold[1])
        ydim = np.arange(hold[0])

# Create 2-D lat/lon arrays from the 1-D arrays
# NOTE: Change shading from 'flat' to 'nearest' for pcolormesh

        x, y = np.meshgrid(xdim,ydim)
        
# Set the plot area

        fig = plt.figure(figsize=(12,6), dpi=120)
        

# Plot the image

        ax1 = fig.add_subplot(1,1,1)
    
        im = ax1.pcolormesh(x,y,img,shading='nearest',cmap=plt.cm.gist_gray,
            vmin=plot_min,vmax=plot_max)

        # Plot the box

        
        ax1.plot([box_x1,box_x2],[box_y1,box_y1],color="lime",linewidth=1) # Bottom
        ax1.plot([box_x2,box_x2],[box_y1,box_y2],color="lime",linewidth=1) # Right
        ax1.plot([box_x1,box_x2],[box_y2,box_y2],color="lime",linewidth=1) # Top
        ax1.plot([box_x1,box_x1],[box_y1,box_y2],color="lime",linewidth=1) # Left
                
        # Set the aspect ratio
                    
        ax1.set_aspect('equal')
        

