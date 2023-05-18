# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:34:55 2023

@author: ULTRASIP_1
"""

#WHATS UP DOLP

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches

#_______________Define ROI Functions___________________________#

def image_crop(a):
        #np.clip(a, 0, None, out=a)
        a[a == -999] = np.nan
        mid_row = a.shape[0] // 2
        mid_col = a.shape[1] // 2
        start_row = mid_row - 262
        end_row = mid_row + 262
        start_col = mid_col - 262
        end_col = mid_col + 262
        
        a = a[start_row:end_row, start_col:end_col]
        return a
    
def calculate_dolp(stokesparam):

    I = stokesparam[0]
    Q = stokesparam[1]
    U = stokesparam[2]
    dolp = ((Q**2 + U**2)**(1/2))/I
    return dolp

    

def calculate_std(image):
# Define the size of the regions we'll calculate the standard deviation for
    region_size = 5

    # Calculate the standard deviation over the regions
    std_dev = np.zeros_like(image)
    for i in range(region_size//2, image.shape[0] - region_size//2):
        for j in range(region_size//2, image.shape[1] - region_size//2):
            std_dev[i,j] = np.std(image[i-region_size//2:i+region_size//2+1, j-region_size//2:j+region_size//2+1])

    return std_dev


def  choose_roi(image): 
            std_dev = calculate_std(image)
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
            
            
            return x,y

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
    datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    #datapath = "C:/Users/ULTRASIP_1/Documents/Bakersfield707_DataCopy/"
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1823/1FIREX"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1823/Bakersfield"
    #outpath = "C:/Users/ULTRASIP_1/Desktop/ForGRASP/Retrieval_Files"


# #Home Computer 
    # datapath = "C:/Users/Clarissa/Documents/AirMSPI/Bakersfield707_Data"
    # outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1823/Bakersfield"

# Load in the set of measurement sequences
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
# Angle Arrays
# ALL ANGLES IN RADIANS
    scat_median = np.zeros((num_step,num_int))  # Scattering angle
    vza_median = np.zeros((num_step,num_meas))  # View zenith angle
    raz_median = np.zeros((num_step,num_meas))  # Relative azimuth angle
    sza_median = np.zeros(num_step)  # Solar zenith angle (one per stare)

#Measurement Arrays   
    i_median = np.zeros((num_step,num_int))  # Intensity
    i_in_polar_median = np.zeros((num_step,num_pol))  # I in polarized bands
    qd_median = np.zeros((num_step,num_pol))  # Qscattering plane
    ud_median = np.zeros((num_step,num_pol))  # Uscattering plane
    q_median = np.zeros((num_step,num_pol))  # Q meridional
    u_median = np.zeros((num_step,num_pol))  # U meridional
    ipol_median = np.zeros((num_step,num_pol))  # Ipol
    dolp_median = np.zeros((num_step,num_pol))  # DoLP
    esd = 0.0  # Earth-Sun distance (only need one)

#Center point Arrays
    center_wave = np.zeros(num_int)  # Center wavelengths  
    center_pol = np.zeros(num_pol)  # Center wavelengths (polarized only)

    
#____________________Sort the data____________________#

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
    
    num_need = num_step*step_ind+num_step
    
    if(num_need > num_files):
        print("***ERROR***")
        print("Not Enough Files for Group Index")
        print(num_need)
        print(num_files)
        print("Check index of the group at Line 44")
        print("***ERROR***")
        print('error')

# Loop through files within the sequence and sort by time (HHMMSS)
# and extract date and target name
#Filenaming strings 
    #Measurement time as an integer
    time_raw = np.zeros((num_files),dtype=int) 
    
    #Time, Date, and Target Name as a string
    time_str_raw = []  
    date_str_raw = []  
    target_str_raw = [] 

#Start the for loop
    for loop in range(num_files):
    
# Select appropriate file

        this_file = raw_list[loop]

# Parse the filename to get information
    
        words = this_file.split('_')
        
        date_str_raw.append(words[4])
        time_str_raw.append(words[5])  # This will retain the "Z" designation
        target_str_raw.append(words[6])
        
        temp = words[5]
        hold = temp.split('Z')
        time_hhmmss = int(hold[0])
        time_raw[loop] = time_hhmmss

# Convert data to numpy arrays

    date_str = np.array(date_str_raw)
    time_str = np.array(time_str_raw)
    target_str = np.array(target_str_raw)

# Sort the files

    sorted = np.argsort(time_raw)
    mspi_list = raw_list[sorted]
    time_list = time_raw[sorted]
    
    date_str_list = date_str[sorted]
    time_str_list = time_str[sorted]
    target_str_list = target_str[sorted]
    
# Loop through the files for one set of step-and-stare acquisitions

    for loop in range(num_step):
    
        this_ind = loop+num_step*step_ind
        
# Test for the middle of the acquisition sequence

        if(loop == mid_step):
            this_date_str = date_str_list[this_ind]
            this_time_str = time_str_list[this_ind]
            this_target_str = target_str_list[this_ind]
                
# Get the filename

        inputName = datapath+'/'+mspi_list[this_ind]
        
# Tell user location in process

        print("Reading: "+inputName)
        
# Open the HDF-5 file

        f = h5py.File(inputName,'r')
        
#_________________________Read the data_________________________#
# Set the datasets and read (355 nm)
# Radiometric Channel

        print("355nm")
        I_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]          
        
# Set the datasets and read (380 nm)
# Radiometric Channel

        print("380nm")
        I_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:]      
        

# Set the datasets and read (445 nm)
# Radiometric Channel

        print("445nm")
        I_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:]     
       
# Set the datasets and read (470 nm)
# Polarized band (INCLUDE SOLAR ANGLES)
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane

        print("470nm")
        I_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]
        DOLP_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/DOLP/'][:]
       
        Qs_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/'][:]
        Us_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/'][:]
        Qm_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_meridian/'][:]
        Um_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_meridian/'][:]
        
        
# Set the datasets and read (555 nm)
# Radiometric Channel

        print("555nm")
        I_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:]     
        
# Set the datasets and read (660 nm)
# Polarized band
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane

        print("660nm")
        I_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:]
        DOLP_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/DOLP/'][:]
       
        Qs_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_scatter/'][:]
        Us_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_scatter/'][:]
        Qm_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_meridian/'][:]
        Um_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_meridian/'][:]
       
      
# Set the datasets and read (865 nm)
# Polarized band
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane
        
        print("865nm")
        I_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:]
        DOLP_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/DOLP/'][:]
        
        Qs_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_scatter/'][:]
        Us_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_scatter/'][:]
        Qm_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_meridian/'][:]
        Um_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_meridian/'][:]
       

# Set the datasets and read (9355 nm)
# Radiometric Channel

        print("935nm")
        I_935 = f['/HDFEOS/GRIDS/935nm_band/Data Fields/I/']      
      
              
# Get the Earth-Sun distance from the file attributes from the first file
        if(esd == 0.0):
            print("GETTING EARTH-SUN DISTANCE")
            esd = f['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/'].attrs['Sun distance']
           
# Get the actual center wavelengths and E0 values
            center_raw = f['/Channel_Information/Center_wavelength/'][:]   
            E0_wave = f['/Channel_Information/Solar_irradiance_at_1_AU/'][:]

# Get the actual center wavelengths and E0 values
            center_raw = f['/Channel_Information/Center_wavelength/'][:]       
            E0_wave = f['/Channel_Information/Solar_irradiance_at_1_AU/'][:]

# Calculate the effective center wavelengths by appropriate averaging
# NOTE: Essentially, for the radiometric only bands, the center wavelength is given in the
#       file. For polarized bands, we average the three available bands.

            center_wave[0] = center_raw[0]  # 355 nm
            center_wave[1] = center_raw[1]  # 380 nm
            center_wave[2] = center_raw[2]  # 445 nm          
            center_wave[3] = (center_raw[3]+center_raw[4]+center_raw[5])/3.0 # 470 nm
            center_wave[4] = center_raw[6]  # 555 nm       
            center_wave[5] = (center_raw[7]+center_raw[8]+center_raw[9])/3.0 # 660 nm
            center_wave[6] = (center_raw[10]+center_raw[11]+center_raw[12])/3.0 # 865 nm
            center_wave[7] = center_raw[13]  # 935 nm          
            
            center_pol[0] = center_wave[3]
            center_pol[1] = center_wave[5]
            center_pol[2] = center_wave[6]
            
# Calculate the effective E0 values by appropriate averaging
# NOTE: Essentially, for radiomentric only bands, the E0 is given in the
#       file. For polarized bands, we average the E0's from the three available bands.

            E0_355 = E0_wave[0]  # 355 nm
            E0_380 = E0_wave[1]  # 380 nm
            E0_445 = E0_wave[2]  # 440 nm
            E0_470 = (E0_wave[3]+E0_wave[4]+E0_wave[5])/3.0 # 470 nm        
            E0_555 = E0_wave[6]  # 555 nm        
            E0_660 = (E0_wave[7]+E0_wave[8]+E0_wave[9])/3.0 # 660 nm
            E0_865 = (E0_wave[10]+E0_wave[11]+E0_wave[12])/3.0 # 865 nm       
            E0_935 = E0_wave[13]  # 935 nm
            
# Get the navigation information if this is the center acquisition
        if(loop == mid_step): #latitude and longitude chosen from nadir of step and stare
            
            print("GETTING NAVIGATION")
                
# Set the datasets and read (Ancillary)
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/']
            elev = dset[:]
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/']
            lat = dset[:]
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/']
            lon = dset[:]
    
# Close the file
        f.close()

#_____________________Perform Data Extraction___________________#
# Extract the data in the large bounding box
# NOTE: This puts the array into *image* space       

        img_i_355 = np.flipud(image_crop(I_355))
        img_i_380 = np.flipud(image_crop(I_380))
        img_i_445 = np.flipud(image_crop(I_445))
        img_i_470 = np.flipud(image_crop(I_470))
        img_i_555 = np.flipud(image_crop(I_555))
        img_i_660 = np.flipud(image_crop(I_660))
        img_i_865 = np.flipud(image_crop(I_865))
        
        img_DOLP_470 = np.flipud(image_crop(DOLP_470))
        img_DOLP_660 = np.flipud(image_crop(DOLP_660))
        img_DOLP_865 = np.flipud(image_crop(DOLP_865))
        
        img_qs_470 = np.flipud(image_crop(Qs_470))
        img_qs_660 = np.flipud(image_crop(Qs_660))
        img_qs_865 = np.flipud(image_crop(Qs_865))
        
        img_us_470 = np.flipud(image_crop(Us_470))
        img_us_660 = np.flipud(image_crop(Us_660))
        img_us_865 = np.flipud(image_crop(Us_865))
        
        img_qm_470 = np.flipud(image_crop(Qm_470))
        img_qm_660 = np.flipud(image_crop(Qm_660))
        img_qm_865 = np.flipud(image_crop(Qm_865))
        
        img_um_470 = np.flipud(image_crop(Um_470))
        img_um_660 = np.flipud(image_crop(Um_660))
        img_um_865 = np.flipud(image_crop(Um_865))
        
        
        
        DOLP_470 = img_DOLP_470[200,200]
        DOLP_660 = img_DOLP_660[200,200]
        DOLP_865 = img_DOLP_865[200,200]
        
        i_470 = img_i_470[200,200]
        i_660 = img_i_660[200,200]
        i_865 = img_i_865[200,200]
 
        qs_470 = img_qs_470[200,200]
        qs_660 = img_qs_660[200,200]
        qs_865 = img_qs_865[200,200]
        us_470 = img_us_470[200,200]
        us_660 = img_us_660[200,200]
        us_865 = img_us_865[200,200]
        
        qm_470 = img_qm_470[200,200]
        qm_660 = img_qm_660[200,200]
        qm_865 = img_qm_865[200,200]
        um_470 = img_um_470[200,200]
        um_660 = img_um_660[200,200]
        um_865 = img_um_865[200,200]
        
        #Input
        #470
        stokesin4 = np.array([[i_470],[qm_470], [um_470]]) #Meridian
        stokesin4s = np.array([[i_470],[qs_470], [us_470]]) #Scattering
        #660
        stokesin6 = np.array([[i_660],[qm_660], [um_660]]) #Meridian
        stokesin6s = np.array([[i_660],[qs_660], [us_660]]) #Scattering
        #865
        stokesin8 = np.array([[i_865],[qm_865], [um_865]]) #Meridian
        stokesin8s = np.array([[i_865],[qs_865], [us_865]]) #Scattering
        
    return stokesin4, stokesin4s, stokesin6, stokesin6s, stokesin8, stokesin8s,DOLP_470,DOLP_660,DOLP_865
        
        
### END MAIN FUNCTION
if __name__ == '__main__':
     merd4,scat4,merd6,scat6,merd8,scat8,dolp4,dolp6,dolp8 = main()     

     dolp4calc = calculate_dolp(merd4)
     dolp4calcscat = calculate_dolp(scat4)