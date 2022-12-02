# -*- coding: utf-8 -*-

# AirMSPI_L1B2_VIS_Prescott_04.py
#
# This is a Python 3.9.13 code to read AirMSPI L1B2 data and make plots.
#
# Creation Date: 2022-07-26
# Last Modified: 2022-07-26
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

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

    basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Aug16_2019_RetrievalFiles/Prescott_Data"
    figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Figures"
    
# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    
# Set the index of the group of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 1
    
# Set some bounds for the image (USER INPUT)

    min_x = 1900
    max_x = 2200
    min_y = 1900
    max_y = 2200
    
# Set some bounds for the sample box (USER INPUT)
# Note: These coordinates are RELATIVE to the overall bounding box

    box_x1 = 120
    box_x2 = 125
    box_y1 = 105
    box_y2 = 110

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

# Set the timer for reading the file

        start_time = time.time()
        
# Open the HDF-5 file

        f = h5py.File(inputName,'r')

# Get the intensity datasets

        dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/']
        I_355 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/']
#        I_380 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/']
#        I_445 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/']
#        I_470 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/']
#        I_555 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/']
#        I_660 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/']
#        I_865 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/935nm_band/Data Fields/I/']
#        I_935 = dset[:]

# Get the polarimetric datasets

#        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/IPOL/']
#        Ipol_470 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/DOLP/']
#        DOLP_470 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/']
#        Q_470 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/']
#        U_470 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/IPOL/']
#        Ipol_660 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/DOLP/']
#        DOLP_660 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/IPOL/']
#        Ipol_865 = dset[:]
    
#        dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/DOLP/']
#        DOLP_865 = dset[:]
    
# Close the file

        f.close()

# Print the time

        end_time = time.time()
        print("Time to Read AirMSPI data was %g seconds" % (end_time - start_time))
        
        print(np.shape(I_355))
        
        img = np.flipud(I_355[min_y:max_y,min_x:max_x])
        
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
        
# Save the data
    
        os.chdir(figpath)

# Generate the output filename
        print(timeoffile_hhmmss)
        outfile = "Prescott_{}".format(loop)
        outfile = outfile+"_"+str(timeoffile_hhmmss)+"_"+str(step_ind)+"v04.png"
        plt.axis('off')
        plt.title("Time: " + str(timeoffile_hhmmss)+" Angle: "+str(angleoffile))
        plt.savefig(outfile,dpi=120)
    
# Print the time

    all_end_time = time.time()
    print("Total elapsed time was %g seconds" % (all_end_time - all_start_time))

# Tell user completion was successful

    print("\nSuccessful Completion\n")

### END MAIN FUNCTION


if __name__ == '__main__':
    main()    