# -*- coding: utf-8 -*-
"""
DataFormatting_AirMSPI_GRASP.py - working version
INPUT: AirMSPI .hdf files
OUTPUT: SDATA structured AirMSPI data products

This is a Python 3.9.13 code to read AirMSPI L1B2 data and 
format the data to perform aerosol retrievals using the 
Generalized Retrieval of Atmosphere and Surface Properties

Code Sections: 
1. Data products
    a. Load in Data
    b. Set ROI 
    c. Sort and Extract Data
    d. Take medians 
2. Geometry Reconciliation 
    a. Put AirMSPI measurements into GRASP geometry 
    b. Normalize radiances
3. Structure the data products according to the GRASP SDATA format 
    a. Third Output File: I in Radiometric Bands (except 935 nm) and I,Q,U in polarized bands
4. Visualize ROI and Measurement Values
    a. Intensity vs Scattering Angle
    b. Q and U vs Scattering Angle
    c. Ipol vs Scattering Angle
    d. Degree of Linear Polarization (DoLP) vs Scattering Angle
    
More info on the GRASP geometry and sdata format can be found at grasp-open.com
and more info on this algoritm can be found in DeLeon et. al. (YYYY)

Creation Date: 2022-08-05
Last Modified: 2023-04-20

by Michael J. Garay and Clarissa M. DeLeon
(Michael.J.Garay@jpl.nasa.gov, cdeleon@arizona.edu)
"""

#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches

#_______________Define ROI Functions___________________________#
def calculate_dolp(stokesparam):

    #I = stokesparam[0]
    Q = stokesparam[0]
    U = stokesparam[1]
    dolp = ((Q**2 + U**2)**(1/2)) #/I
    return dolp

def image_crop(a):
        #np.clip(a, 0, None, out=a)
        # a[a == -999] = np.nan
        # a = a[~np.isnan(a).all(axis=1), :]
        # a = a[~np.isnan(a).all(axis=1)]
        
        a = a[~(a== -999).all(axis=1)]
        a = a[:,~(a== -999).all(axis=0)]
        a[np.where(a == -999)] = np.nan


        mid_row = a.shape[0] // 2
        mid_col = a.shape[1] // 2
        start_row = mid_row - 262
        end_row = mid_row + 262
        start_col = mid_col - 262
        end_col = mid_col + 262
        
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
    #datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    #datapath = "C:/Users/ULTRASIP_1/Documents/Pinehurst/"
    #datapath = "C:/Users/ULTRASIP_1/Documents/Bakersfield_0708/"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May2523/2FIREX"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May2523/Bakersfield_2"
    #outpath = "C:/Users/ULTRASIP_1/Desktop/ForGRASP/Retrieval_Files"
    datapath = "C:/Users/ULTRASIP_1/Documents/Inchelium"
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/June2523/Washington1"



# # #Home Computer 
    #datapath = "C:/Users/Clarissa/Documents/AirMSPI/Washington"
#     datapath = "C:/Users/Clarissa/Documents/AirMSPI/Prescott/FIREX-AQ_8172019"

    #outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/June2523/Washington1"
#     outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1923/1FIREX"
# # Load in the set of measurement sequences
# Set the length of one measurement sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 9
    
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
        scat_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/Scattering_angle/'][:] 
        vaz_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_azimuth/'][:]       
        vza_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_zenith/'][:]
        print(type(I_355))
# Set the datasets and read (380 nm)
# Radiometric Channel

        print("380nm")
        I_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:]      
        scat_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/Scattering_angle/'][:] 
        vaz_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_azimuth/'][:]
        vza_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_zenith/'][:]

# Set the datasets and read (445 nm)
# Radiometric Channel

        print("445nm")
        I_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:]     
        scat_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/Scattering_angle/'][:] 
        vaz_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_azimuth/'][:]
        vza_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_zenith/'][:]
        
# Set the datasets and read (470 nm)
# Polarized band (INCLUDE SOLAR ANGLES)
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane

        print("470nm")
        I_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]
        DOLP_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/DOLP/'][:]
        IPOL_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/IPOL/'][:]
        scat_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Scattering_angle/'][:]
        saz_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Sun_azimuth/'][:]
        sza_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Sun_zenith/'][:]
        Qs_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/'][:]
        Us_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/'][:]
        Qm_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_meridian/'][:]
        Um_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_meridian/'][:]
        vaz_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_azimuth/'][:]
        vza_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_zenith/'][:]

        
# Set the datasets and read (555 nm)
# Radiometric Channel

        print("555nm")
        I_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:]     
        scat_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/Scattering_angle/'][:] 
        vaz_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_azimuth/'][:]
        vza_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_zenith/'][:]
                
# Set the datasets and read (660 nm)
# Polarized band
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane

        print("660nm")
        I_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:]
        DOLP_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/DOLP/'][:]
        IPOL_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/IPOL/'][:]
        scat_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Scattering_angle/'][:]
        saz_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Sun_azimuth/'][:]
        sza_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Sun_zenith/'][:]
        Qs_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_scatter/'][:]
        Us_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_scatter/'][:]
        Qm_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_meridian/'][:]
        Um_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_meridian/'][:]
        vaz_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_azimuth/'][:]
        vza_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_zenith/'][:]
      
# Set the datasets and read (865 nm)
# Polarized band
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane
        
        print("865nm")
        I_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:]
        DOLP_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/DOLP/'][:]
        IPOL_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/IPOL/'][:]
        scat_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Scattering_angle/'][:]
        saz_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Sun_azimuth/'][:]
        sza_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Sun_zenith/'][:]
        Qs_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_scatter/'][:]
        Us_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_scatter/'][:]
        Qm_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_meridian/'][:]
        Um_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_meridian/'][:]
        vaz_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_azimuth/'][:]
        vza_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_zenith/'][:]

# Set the datasets and read (9355 nm)
# Radiometric Channel

        print("935nm")
        I_935 = f['/HDFEOS/GRIDS/935nm_band/Data Fields/I/']      
        scat_935 = f['/HDFEOS/GRIDS/935nm_band/Data Fields/Scattering_angle/'] 
        vaz_935 = f['/HDFEOS/GRIDS/935nm_band/Data Fields/View_azimuth/']
        vza_935 = f['/HDFEOS/GRIDS/935nm_band/Data Fields/View_zenith/']
              
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
           
        img_scat_355 = np.flipud(image_crop(scat_355))
        img_scat_380 = np.flipud(image_crop(scat_380))
        img_scat_445 = np.flipud(image_crop(scat_445))
        img_scat_470 = np.flipud(image_crop(scat_470))
        img_scat_555 = np.flipud(image_crop(scat_555))
        img_scat_660 = np.flipud(image_crop(scat_660))
        img_scat_865 = np.flipud(image_crop(scat_865))
        
        img_vaz_355 = np.flipud(image_crop(vaz_355))
        img_vaz_380 = np.flipud(image_crop(vaz_380))
        img_vaz_445 = np.flipud(image_crop(vaz_445))
        img_vaz_470 = np.flipud(image_crop(vaz_470))
        img_vaz_555 = np.flipud(image_crop(vaz_555))
        img_vaz_660 = np.flipud(image_crop(vaz_660))
        img_vaz_865 = np.flipud(image_crop(vaz_865))
        
        img_vza_355 = np.flipud(image_crop(vza_355))
        img_vza_380 = np.flipud(image_crop(vza_380))
        img_vza_445 = np.flipud(image_crop(vza_445))
        img_vza_470 = np.flipud(image_crop(vza_470))
        img_vza_555 = np.flipud(image_crop(vza_555))
        img_vza_660 = np.flipud(image_crop(vza_660))
        img_vza_865 = np.flipud(image_crop(vza_865))
        
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
        
        
        img_saz = np.flipud(image_crop(saz_470))
        img_sza = np.flipud(image_crop(sza_470))
        
#If this is the center acquisition, process the navigation information

        if(loop == mid_step):
            img_lat = np.flipud(image_crop(lat))
            img_lon = np.flipud(image_crop(lon))
            img_elev = np.flipud(image_crop(elev))
        
# Test for valid data    
# NOTE: The test is done for all the wavelengths that are read, so if the wavelengths
#       are changed, then the test needs to change  - note about missing data  
        
        good = ((img_i_355 > 0.0) & (img_i_380 > 0.0) & (img_i_445 > 0.0) &
            (img_i_470 > 0.0) & (img_i_555 > 0.0) & (img_i_660 > 0.0) &
            (img_i_865 > 0.0))
            
        img_good = img_i_355[good]
        
        if(len(img_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print('error')
         
     
        
        if loop == 0: 
            plt.figure()
            plt.imshow(I_470)
            plt.figure()
            plt.imshow(img_i_470)
            x_center, y_center = choose_roi(img_i_470)
        else:
            plt.figure()
            plt.imshow(I_470)
            plt.title('I_470')
            plt.figure()
            plt.imshow(img_i_470)
        
        box_x1 = x_center - 2 
        box_x2 = x_center + 3 
        box_y1 = y_center - 2 
        box_y2 = y_center + 3 
                    
# Extract the values from the ROI
# NOTE: The coordinates are relative to the flipped "img" array

        box_i_355 = img_i_355[box_x1:box_x2,box_y1:box_y2]
        box_i_380 = img_i_380[box_x1:box_x2,box_y1:box_y2]
        box_i_445 = img_i_445[box_x1:box_x2,box_y1:box_y2]
        box_i_470 = img_i_470[box_x1:box_x2,box_y1:box_y2]
        box_i_555 = img_i_555[box_x1:box_x2,box_y1:box_y2]
        box_i_660 = img_i_660[box_x1:box_x2,box_y1:box_y2]
        box_i_865 = img_i_865[box_x1:box_x2,box_y1:box_y2]
        
        box_scat_355 = img_scat_355[box_x1:box_x2,box_y1:box_y2]
        box_scat_380 = img_scat_380[box_x1:box_x2,box_y1:box_y2]
        box_scat_445 = img_scat_445[box_x1:box_x2,box_y1:box_y2]
        box_scat_470 = img_scat_470[box_x1:box_x2,box_y1:box_y2]
        box_scat_555 = img_scat_555[box_x1:box_x2,box_y1:box_y2]
        box_scat_660 = img_scat_660[box_x1:box_x2,box_y1:box_y2]
        box_scat_865 = img_scat_865[box_x1:box_x2,box_y1:box_y2]
        
        box_vaz_355 = img_vaz_355[box_x1:box_x2,box_y1:box_y2]
        box_vaz_380 = img_vaz_380[box_x1:box_x2,box_y1:box_y2]
        box_vaz_445 = img_vaz_445[box_x1:box_x2,box_y1:box_y2]
        box_vaz_470 = img_vaz_470[box_x1:box_x2,box_y1:box_y2]
        box_vaz_555 = img_vaz_555[box_x1:box_x2,box_y1:box_y2]
        box_vaz_660 = img_vaz_660[box_x1:box_x2,box_y1:box_y2]
        box_vaz_865 = img_vaz_865[box_x1:box_x2,box_y1:box_y2]
        
        box_vza_355 = img_vza_355[box_x1:box_x2,box_y1:box_y2]
        box_vza_380 = img_vza_380[box_x1:box_x2,box_y1:box_y2]
        box_vza_445 = img_vza_445[box_x1:box_x2,box_y1:box_y2]
        box_vza_470 = img_vza_470[box_x1:box_x2,box_y1:box_y2]
        box_vza_555 = img_vza_555[box_x1:box_x2,box_y1:box_y2]
        box_vza_660 = img_vza_660[box_x1:box_x2,box_y1:box_y2]
        box_vza_865 = img_vza_865[box_x1:box_x2,box_y1:box_y2]
        
        box_qs_470 = img_qs_470[box_x1:box_x2,box_y1:box_y2]
        box_qs_660 = img_qs_660[box_x1:box_x2,box_y1:box_y2]
        box_qs_865 = img_qs_865[box_x1:box_x2,box_y1:box_y2]
        
        
        box_us_470 = img_us_470[box_x1:box_x2,box_y1:box_y2]
        box_us_660 = img_us_660[box_x1:box_x2,box_y1:box_y2]
        box_us_865 = img_us_865[box_x1:box_x2,box_y1:box_y2]
        

        box_qm_470 = img_qm_470[box_x1:box_x2,box_y1:box_y2]
        box_qm_660 = img_qm_660[box_x1:box_x2,box_y1:box_y2]
        box_qm_865 = img_qm_865[box_x1:box_x2,box_y1:box_y2]
        
        
        box_um_470 = img_um_470[box_x1:box_x2,box_y1:box_y2]
        box_um_660 = img_um_660[box_x1:box_x2,box_y1:box_y2]
        box_um_865 = img_um_865[box_x1:box_x2,box_y1:box_y2]

        box_saz = img_saz[box_x1:box_x2,box_y1:box_y2]
        box_sza = img_sza[box_x1:box_x2,box_y1:box_y2]
        
# If this is the center acquisition, process the navigation information

        if(loop == mid_step):
            box_lat = img_lat[box_x1:box_x2,box_y1:box_y2]
            box_lon = img_lon[box_x1:box_x2,box_y1:box_y2]
            box_elev = img_elev[box_x1:box_x2,box_y1:box_y2]
        
# Extract the valid data and calculate the median
# NOTE: The test is done for all the wavelengths that are read, so if the wavelengths
#       are changed, then the test needs to change    
        
        good = ((box_i_355 > 0.0) & (box_i_380 > 0.0) & (box_i_445 > 0.0) &
            (box_i_470 > 0.0) & (box_i_555 > 0.0) & (box_i_660 > 0.0) &
            (box_i_865 > 0.0))
            
        box_good = box_i_355[good]
        
        plt.imshow(box_i_470, cmap = 'gray')
        print(box_i_470[good])
        
        if(len(box_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print('error')

        i_355 = np.median(box_i_355[good])
        idx_355 = 11 #np.where(box_i_355[good] == i_355)[0]
        print("idk is:",idx_355)
        
        i_380 = box_i_380[good][idx_355]
        
        # np.median(box_i_380[good])
        # idx_380 = np.where(box_i_380[good] == i_380)[0]
        # print("idk is:",idx_380)
        
        i_445 = box_i_445[good][idx_355]
        # np.median(box_i_445[good])
        # idx_445 = np.where(box_i_445[good] == i_445)[0]
        # print("idk is:",idx_445)
        
        i_470 = box_i_470[good][idx_355]
        # np.median(box_i_470[good])
        # idx_470 = np.where(box_i_470[good] == i_470)[0]
        # print("idk is:",idx_470)
        
        i_555 = box_i_555[good][idx_355]
        # np.median(box_i_555[good])
        # idx_555 = np.where(box_i_555[good] == i_555)[0]
        # print("idk is:",idx_555)
        
        i_660 = box_i_660[good][idx_355]
        # np.median(box_i_660[good])
        # idx_660 = np.where(box_i_660[good] == i_660)[0]
        # print("idk is:",idx_660)
        
        i_865 = box_i_865[good][idx_355]
        # np.median(box_i_865[good])
        # idx_865 = np.where(box_i_865[good] == i_865)[0]
        # print("idk is:",idx_865)
           
        vaz_355 = box_vaz_355[good][idx_355]
        #np.median(box_vaz_355[good])
        vaz_380 = box_vaz_380[good][idx_355]
        #np.median(box_vaz_380[good])
        vaz_445 = box_vaz_445[good][idx_355]
        #np.median(box_vaz_445[good])
        vaz_470 = box_vaz_470[good][idx_355]
        #np.median(box_vaz_470[good])
        vaz_555 = box_vaz_555[good][idx_355]
        #np.median(box_vaz_555[good])
        vaz_660 = box_vaz_660[good][idx_355]
        #np.median(box_vaz_660[good])
        vaz_865 = box_vaz_865[good][idx_355]
        #np.median(box_vaz_865[good])
        
        vza_355 = box_vza_380[good][idx_355]
        #np.median(box_vza_355[good])
        vza_380 = box_vza_380[good][idx_355]
        #np.median(box_vza_380[good])
        vza_445 = box_vza_445[good][idx_355]
        #np.median(box_vza_445[good])
        vza_470 = box_vza_470[good][idx_355]
        #np.median(box_vza_470[good])
        vza_555 = box_vza_555[good][idx_355]
        #np.median(box_vza_555[good])
        vza_660 = box_vza_660[good][idx_355]
        #np.median(box_vza_660[good])
        vza_865 = box_vza_865[good][idx_355]
        #np.median(box_vza_865[good])
        
        qs_470 = box_qs_470[good][idx_355]
        #np.median(box_qs_470[good])
        qs_660 = box_qs_660[good][idx_355]
        #np.median(box_qs_660[good])
        qs_865 = box_qs_865[good][idx_355]
        #np.median(box_qs_865[good])
        
        us_470 = box_us_470[good][idx_355]
        #np.median(box_us_470[good])
        us_660 = box_us_660[good][idx_355]
        #np.median(box_us_660[good])
        us_865 = box_us_865[good][idx_355]
        #np.median(box_us_865[good])
        
        qm_470 = box_qm_470[good][idx_355]
        qm_660 = box_qm_660[good][idx_355]
        qm_865 = box_qm_865[good][idx_355]

        um_470 = box_um_470[good][idx_355]
        um_660 = box_um_660[good][idx_355]
        um_865 = box_um_865[good][idx_355]
        
     
        # qm_470 = np.median(box_qm_470[good])
        # qm_660 = np.median(box_qm_660[good])
        # qm_865 = np.median(box_qm_865[good])
        
        
        # um_470 = np.median(box_um_470[good])
        # um_660 = np.median(box_um_660[good])
        # um_865 = np.median(box_um_865[good])


        # saz = np.median(box_saz[good])
        # sza = np.median(box_sza[good])
        saz = box_saz[good][idx_355]
        sza = box_sza[good][idx_355]
        
        
# If this is the center acquisition, process the navigation information

        if(loop == mid_step):
            lat_median = np.median(box_lat[good])
            lon_median = np.median(box_lon[good])
            elev_median = np.median(box_elev[good])
            if(elev_median < 0.0):
                elev_median = 0.0  # Do not allow negative elevation

#________________________Section 2: Geometry Reconciliation___________________________#

        #Input
        #470
        stokesin4 = np.array([[qm_470], [um_470]]) #Meridian
        stokesin4s = np.array([[qs_470], [us_470]]) #Scattering
        #660
        stokesin6 = np.array([[qm_660], [um_660]]) #Meridian
        stokesin6s = np.array([[qs_660], [us_660]]) #Scattering
        #865
        stokesin8 = np.array([[qm_865], [um_865]]) #Meridian
        stokesin8s = np.array([[qs_865], [us_865]]) #Scattering
        

        
        # qg_470, ug_470 = stokesin4s
        # qg_660, ug_660 = stokesin6s
        # qg_865, ug_865 = stokesin8s
        
        qg_470, ug_470 = stokesin4
        qg_660, ug_660 = stokesin6
        qg_865, ug_865 = stokesin8

        


        if saz >= 180: 
            saz = saz - 180
        else:
            saz = saz + 180

        
        raz_355 = saz - vaz_355
        raz_380 = saz - vaz_380
        raz_445 = saz - vaz_445
        raz_470 = saz - vaz_470
        raz_555 = saz - vaz_555
        raz_660 = saz - vaz_660
        raz_865 = saz - vaz_865
        
        if raz_355 < 0:
            raz_355 = raz_355 + 360
        if raz_380 < 0:
            raz_380 = raz_380 + 360
        if raz_445 < 0:
            raz_445 = raz_445 + 360
        if raz_470 < 0:
            raz_470 = raz_470 + 360
        if raz_555 < 0:
            raz_555 = raz_555 + 360
        if raz_660 < 0:
            raz_660 = raz_660 + 360
        if raz_865 < 0:
            raz_865 = raz_865 + 360        
      
        
## NORMALIZE THE RADIANCES TO THE median EARTH-SUN DISTANCE AND CONVERT TO 
### EQUIVALENT REFLECTANCES = PI*L/E0

        eqr_i_355 = np.pi*i_355*esd**2/E0_355
        eqr_i_380 = np.pi*i_380*esd**2/E0_380
        eqr_i_445 = np.pi*i_445*esd**2/E0_445
        eqr_i_470 = np.pi*i_470*esd**2/E0_470
        eqr_i_555 = np.pi*i_555*esd**2/E0_555
        eqr_i_660 = np.pi*i_660*esd**2/E0_660
        eqr_i_865 = np.pi*i_865*esd**2/E0_865
        
        eqr_qg_470 = np.pi*qg_470*esd**2/E0_470
        eqr_qg_660 = np.pi*qg_660*esd**2/E0_660
        eqr_qg_865 = np.pi*qg_865*esd**2/E0_865
        
        eqr_ug_470 = np.pi*ug_470*esd**2/E0_470
        eqr_ug_660 = np.pi*ug_660*esd**2/E0_660
        eqr_ug_865 = np.pi*ug_865*esd**2/E0_865
        
        # print('dolp:',calculate_dolp(stokesin4))
        # print('dolps:',calculate_dolp(stokesin4s))
        

#____________________________STORE THE DATA____________________________#

        i_median[loop,0] = eqr_i_355
        i_median[loop,1] = eqr_i_380
        i_median[loop,2] = eqr_i_445
        i_median[loop,3] = eqr_i_470
        i_median[loop,4] = eqr_i_555
        i_median[loop,5] = eqr_i_660
        i_median[loop,6] = eqr_i_865
                
        
        
        vza_median[loop,0] = vza_355
        vza_median[loop,1] = vza_380
        vza_median[loop,2] = vza_445
        vza_median[loop,3] = vza_470
        vza_median[loop,4] = vza_470
        vza_median[loop,5] = vza_470
        vza_median[loop,6] = vza_555
        vza_median[loop,7] = vza_660
        vza_median[loop,8] = vza_660
        vza_median[loop,9] = vza_660
        vza_median[loop,10] = vza_865
        vza_median[loop,11] = vza_865
        vza_median[loop,12] = vza_865
        
        raz_median[loop,0] = raz_355
        raz_median[loop,1] = raz_380
        raz_median[loop,2] = raz_445
        raz_median[loop,3] = raz_470
        raz_median[loop,4] = raz_470
        raz_median[loop,5] = raz_470
        raz_median[loop,6] = raz_555
        raz_median[loop,7] = raz_660
        raz_median[loop,8] = raz_660
        raz_median[loop,9] = raz_660
        raz_median[loop,10] = raz_865
        raz_median[loop,11] = raz_865
        raz_median[loop,12] = raz_865

        q_median[loop,0] = eqr_qg_470
        q_median[loop,1] = eqr_qg_660
        q_median[loop,2] = eqr_qg_865
    
        u_median[loop,0] = eqr_ug_470
        u_median[loop,1] = eqr_ug_660
        u_median[loop,2] = eqr_ug_865

        # q_median[loop,0] = -eqr_qg_470
        # q_median[loop,1] = -eqr_qg_660
        # q_median[loop,2] = -eqr_qg_865
    
        # u_median[loop,0] = -eqr_ug_470
        # u_median[loop,1] = -eqr_ug_660
        # u_median[loop,2] = -eqr_ug_865
        

        sza_median[loop] = sza


# #__________________Section 3: Output Data in GRASP SDATA Format__________________#
# Guide to output file names
# NOTE: The options more or less correspond to GRASP retrieval.regime_of_measurement_fitting
#       0 = .radiance (option 1)
#       1-5 = .polarization (option as given)


# Change to the output directory
    os.chdir(outpath) 
    
# Generate the base output file name
    #outfile_base = "AirMSPI_"+this_date_str+"_"+this_time_str+"_"
    #outfile_base = outfile_base+this_target_str+"_"
    outfile_base = 'RotfromMerd1'

# Get the software version number to help track issues
    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0]

### THIRD OUTPUT FILE: I IN SPECTRAL BANDS AND  I, Q, U IN POLARIZED BANDS
    num_intensity = 7
    num_polar = 3
    num_all = num_intensity+num_polar
        
# Generate an output file name

    outfile = outfile_base+".sdat"
        
    print()
    print("Saving: "+outfile)
    
# Open the output file

    outputFile = open(outfile, 'w')
        
# Write the sdat header information

    out_str = 'SDATA version 2.0\n'
    outputFile.write(out_str)
    out_str = '  1   1   1  : NX NY NT\n'
    outputFile.write(out_str)
    out_str = '\n'
    outputFile.write(out_str)

# Parse the date string into the correct format

    sdat_date = this_date_str[0:4]+'-'+this_date_str[4:6]+'-'+this_date_str[6:8]
    print(sdat_date)
        
# Parse the time string into the correct format

    sdat_time = this_time_str[0:2]+':'+this_time_str[2:4]+':'+this_time_str[4:7]
    print(sdat_time)
        
# Write out the data header line

    out_str = '  1   '+sdat_date+'T'+sdat_time
    out_str = out_str+'       70000.00   0   1   : NPIXELS  TIMESTAMP  HEIGHT_OBS(m)  NSURF  IFGAS    1\n'
    outputFile.write(out_str)
    
# Generate content for sdat (single line)

    out_str = '           1'  # x-coordinate (ix)
    out_str = out_str+'           1'  # y-coordinate (iy)
    out_str = out_str+'           1'  # Cloud Flag (0=cloud, 1=clear)
    out_str = out_str+'           1'  # Pixel column in grid (icol)
    out_str = out_str+'           1'  # Pixel line in grid (row)

    out_str = out_str+'{:19.8f}'.format(lon_median)  # Longitude
    out_str = out_str+'{:18.8f}'.format(lat_median)  # Latitude
    out_str = out_str+'{:17.8f}'.format(elev_median) # Elevation

    out_str = out_str+'      100.000000'  # Percent of land
    out_str = out_str+'{:16d}'.format(num_intensity)  # Number of wavelengths (nwl)
    
  ## SET UP THE WAVELENGTH AND MEASUREMENT INFORMATION
        
# Loop through wavelengths

    for loop in range(num_intensity):
        out_str = out_str+'{:17.9f}'.format(center_wave[loop]/1000.)  # Wavelengths in microns
       
    # Loop over the number of types of measurements per wavelength

# for loop in range(num_intensity):
    out_str = out_str+'{:12d}'.format(1)
    out_str = out_str+'{:12d}'.format(1) # 1 measurement per wavelength
    out_str = out_str+'{:12d}'.format(1)
    out_str = out_str+'{:12d}'.format(3)
    out_str = out_str+'{:12d}'.format(1)
    out_str = out_str+'{:12d}'.format(3)
    out_str = out_str+'{:12d}'.format(3)

# Loop over the measurement types per wavelength
# NOTE: Values can be found in the GRASP documentation in Table 4.5
#       41 = Normalized radiance (I = rad*pi/E0) - GRASP calls normalized (reduced) radiance

    #for loop in range(num_intensity):
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(42)
    out_str = out_str+'{:12d}'.format(43)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(42)
    out_str = out_str+'{:12d}'.format(43)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(42)
    out_str = out_str+'{:12d}'.format(43)


        
# Loop over the number of measurements per wavelength
# Note: This is the number of stares in the step-and-stare sequence

    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)
    out_str = out_str+'{:12d}'.format(num_step)

## ANGLE DEFINITIONS

# Solar zenith angle per wavelength
# NOTE: This is per wavelength rather than per measurement (probably because of 
#       AERONET), so we take the average solar zenith angle, although this
#       varies from measurement to measurement from AirMSPI

    sza_median = np.median(sza_median)

    for loop in range(num_intensity):
        out_str = out_str+'{:16.8f}'.format(sza_median)

    for outer in range(num_meas):
        for inner in range(num_step): 
            out_str = out_str+'{:16.8f}'.format(vza_median[inner,outer])


# Relative azimuth angle per measurement per wavelength
    for outer in range(num_meas):
        for inner in range(num_step): 
            out_str = out_str+'{:16.8f}'.format(raz_median[inner,outer])


#Measurements
    for outer in [0,1,2]:  # Loop over wavelengths
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])
    
    for outer in [3]:  # Loop over wavelengths
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
        # for inner in range(num_step):  # Loop over measurements
        #     out_str = out_str+'{:16.8f}'.format(i_in_polar_median[inner,0])  # Ipol
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(q_median[inner,0])  # Q
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(u_median[inner,0])  # U

    for outer in [4]:  # Loop over wavelengths
        for inner in range(num_step):  # Loop over measurements
                out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])

    for outer in [5]:  # Loop over wavelengths
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(q_median[inner,1])  # Q
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(u_median[inner,1])  # U


    for outer in [6]:  # Loop over wavelengths
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(q_median[inner,2])  # Q
        for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(u_median[inner,2])  # U
            
            
## ADDITIONAL PARAMETERS
# NOTE: This is kludgy and GRASP seems to run without this being entirely correct

    out_str = out_str+'       0.00000000'  # Ground parameter (wave 1)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 2)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 3)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 4)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 5)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 6)
    out_str = out_str+'       0.00000000'  # Ground parameter (wave 7)
    out_str = out_str+'       0'  # Gas parameter (wave 1)
    out_str = out_str+'       0'  # Gas parameter (wave 2)
    out_str = out_str+'       0'  # Gas parameter (wave 3)
    out_str = out_str+'       0'  # Gas parameter (wave 4)
    out_str = out_str+'       0'  # Gas parameter (wave 5)
    out_str = out_str+'       0'  # Gas parameter (wave 6)
    out_str = out_str+'       0'  # Gas parameter (wave 7)
    out_str = out_str+'       0'  # Covariance matrix (wave 1)
    out_str = out_str+'       0'  # Covariance matrix (wave 2)
    out_str = out_str+'       0'  # Covariance matrix (wave 3)
    out_str = out_str+'       0'  # Covariance matrix (wave 4)
    out_str = out_str+'       0'  # Covariance matrix (wave 5)
    out_str = out_str+'       0'  # Covariance matrix (wave 6)
    out_str = out_str+'       0'  # Covariance matrix (wave 7)
    out_str = out_str+'       0'  # Vertical profile (wave 1)
    out_str = out_str+'       0'  # Vertical profile (wave 2)
    out_str = out_str+'       0'  # Vertical profile (wave 3)
    out_str = out_str+'       0'  # Vertical profile (wave 4)
    out_str = out_str+'       0'  # Vertical profile (wave 5)
    out_str = out_str+'       0'  # Vertical profile (wave 6)
    out_str = out_str+'       0'  # Vertical profile (wave 7)
    out_str = out_str+'       0'  # (Dummy) (wave 1)
    out_str = out_str+'       0'  # (Dummy) (wave 2)
    out_str = out_str+'       0'  # (Dummy) (wave 3)
    out_str = out_str+'       0'  # (Dummy) (wave 4)
    out_str = out_str+'       0'  # (Dummy) (wave 5)
    out_str = out_str+'       0'  # (Dummy) (wave 6)
    out_str = out_str+'       0'  # (Dummy) (wave 7)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
    out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
                   
# # Endline
       
    out_str = out_str+'\n'

# Write out the line
     
    outputFile.write(out_str)

# Close the output file

    outputFile.close()        
        

### END MAIN FUNCTION
if __name__ == '__main__':
     main() 
