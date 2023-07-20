# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:40:07 2023

@author: ULTRASIP_1
"""

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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# -*- coding: utf-8 -*-
"""
Functions
"""

def calculate_dolp(stokesparam):

    #I = stokesparam[0]
    Q = stokesparam[0]
    U = stokesparam[1]
    dolp = ((Q**2 + U**2)**(1/2)) #/I
    return dolp

def image_crop(a):
        a = a[~(a== -999).all(axis=1)]
        a = a[:,~(a== -999).all(axis=0)]
        a[np.where(a == -999)] = np.nan
        
        mid_row = a.shape[0] // 2
        mid_col = a.shape[1] // 2
        start_row = mid_row - 1048
        end_row = mid_row +1048
        start_col = mid_col - 1048
        end_col = mid_col + 1048
        
        a = a[start_row:end_row, start_col:end_col]
        return a





def calculate_std(image):
# Define the size of the regions we'll calculate the standard deviation for
    region_size = 5

    # Calculate the standard deviation over the regions
    std_dev = np.zeros_like(image)
    # for i in range(region_size//2, image.shape[0] - region_size//2):
    #     for j in range(region_size//2, image.shape[1] - region_size//2):
    #         std_dev[i,j] = np.std(image[i-region_size//2:i+region_size//2+1, j-region_size//2:j+region_size//2+1])

    return std_dev


def  choose_roi(latitude,longitude,image): 
            #std_dev = plot_radiance(latitude,longitude,image)
    # Plot the original image and the standard deviation image side by side
            fig, ax = plt.subplots(1,2,  figsize=(16, 8))
            ax[0].imshow(image , cmap = 'gray')
            ax[0].set_title('Original Image')
            ax[0].grid(True)

            ax[1].set_title('Plot of Lat/Long')
            ax[1].scatter(longitude, latitude, c=image, cmap='jet', s=5)
            # Add more grid lines
            plt.gca().xaxis.set_major_locator(MultipleLocator(0.001))
            plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
            
            # Show every third label on the x-axis
            plt.gca().xaxis.set_major_locator(MultipleLocator(base=0.01))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            ax[1].grid(True)
            
            plt.show()

        # Prompt the user to choose a region
            x = int(input('Enter x-coordinate of region: '))
            y = int(input('Enter y-coordinate of region: '))
                        
            
            return x,y
        
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



#_______________Start the main code____________#
def main():  # Main code

#___________________Section 1: Data Products______________________#


#Load in Data
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/ or 
#

# Set paths to AirMSPI data and where the output SDATA file should be saved 
# NOTE: datapath is the location of the AirMSPI HDF data files
#       outpath is where the output should be stored


    datapath = format_directory()
    outpath = format_directory()
    pol_ref_plane = input("Enter polarimetric ref plane (Scattering or Meridian): ")

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
    num_meas = 13
# Create arrays to store data
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
    #Time, Date, and Target Name as a string
    time_str_raw = []  
    date_str_raw = []  
    target_str_raw = [] 
#Center point Arrays
    center_wave = np.zeros(num_int)  # Center wavelengths  
    center_pol = np.zeros(num_pol)  # Center wavelengths (polarized only)
# Load in data files

    data_files = load_hdf_files(datapath, num_step,step_ind)
    
# Loop through the files for one set of step-and-stare acquisitions

    for loop in range(num_step):
    
        file_to_read = data_files[loop]
# Tell user location in process

        print("Reading: "+file_to_read)
        
# Open the HDF-5 file

        f = h5py.File(file_to_read,'r')
        
#_________________________Read the data_________________________#
# Set the datasets and read (355 nm)
# Radiometric Channel

        print("355nm")
        I_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]          
        scat_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/Scattering_angle/'][:] 
        vaz_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_azimuth/'][:]       
        vza_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_zenith/'][:]
        
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
        
# Get the Earth-Sun distance from the file attributes from the first file
        if(esd == 0.0):
            print("GETTING EARTH-SUN DISTANCE")
            esd = f['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/'].attrs['Sun distance']
           
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
        if(loop == 0): #latitude and longitude chosen from nadir of step and stare
            
            print("GETTING NAVIGATION AND DATE/TIME")
                
# Set the datasets and read (Ancillary)
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/']
            elev = dset[:]
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/']
            lat = dset[:]
            dset = f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/']
            lon = dset[:]
            words = file_to_read.split('_')
            print(words)
            
            date_str_raw.append(words[5])
            print(date_str_raw)
            time_str_raw.append(words[6])  # This will retain the "Z" designation
            target_str_raw.append(words[7])
            
            # temp = words[7]
            # hold = temp.split('Z')
            # time_hhmmss = int(hold[0])
            # time_str_raw = time_hhmmss

    # Convert data to numpy arrays

            date_str = np.array(date_str_raw)
            time_str = np.array(time_str_raw)
            target_str = np.array(target_str_raw)
# Close the file
        f.close()

#_____________________Perform Data Extraction___________________#
# Extract the data in the large bounding box
# NOTE: This puts the array into *image* space       

        img_i_355 = image_crop(I_355)
        img_i_380 = image_crop(I_380)
        img_i_445 = (image_crop(I_445))
        img_i_470 = (image_crop(I_470))
        img_i_555 = (image_crop(I_555))
        img_i_660 = (image_crop(I_660))
        img_i_865 = (image_crop(I_865))
        
        img_vaz_355 = image_crop(vaz_355)
        img_vaz_380 = image_crop(vaz_380)
        img_vaz_445 = image_crop(vaz_445)
        img_vaz_470 = image_crop(vaz_470)
        img_vaz_555 = image_crop(vaz_555)
        img_vaz_660 = image_crop(vaz_660)
        img_vaz_865 = image_crop(vaz_865)
        
        img_vza_355 = image_crop(vza_355)
        img_vza_380 = (image_crop(vza_380))
        img_vza_445 = (image_crop(vza_445))
        img_vza_470 = (image_crop(vza_470))
        img_vza_555 = (image_crop(vza_555))
        img_vza_660 = (image_crop(vza_660))
        img_vza_865 = (image_crop(vza_865))
        
        img_qs_470 = (image_crop(Qs_470))
        img_qs_660 = (image_crop(Qs_660))
        img_qs_865 = (image_crop(Qs_865))
        
        img_us_470 = (image_crop(Us_470))
        img_us_660 = (image_crop(Us_660))
        img_us_865 = (image_crop(Us_865))
        
        img_qm_470 = (image_crop(Qm_470))
        img_qm_660 = (image_crop(Qm_660))
        img_qm_865 = (image_crop(Qm_865))
        
        img_um_470 = (image_crop(Um_470))
        img_um_660 = (image_crop(Um_660))
        img_um_865 = (image_crop(Um_865))
        
        
        img_saz = (image_crop(saz_470))
        img_sza = (image_crop(sza_470))
        
#If this is the center acquisition, process the navigation information

        if(loop == 0):
            img_lat = (image_crop(lat))
            img_lon = (image_crop(lon))
            img_elev = (image_crop(elev))
        
# Test for valid data    
# NOTE: The test is done for all the wavelengths that are read, so if the wavelengths
#       are changed, then the test needs to change  - note about missing data  
        
        # good = ((img_i_355 > 0.0) & (img_i_380 > 0.0) & (img_i_445 > 0.0) &
        #     (img_i_470 > 0.0) & (img_i_555 > 0.0) & (img_i_660 > 0.0) &
        #     (img_i_865 > 0.0))
            
        # img_good = img_i_355[good]
        
        # if(len(img_good) < 1):
        #     print("***ERROR***")
        #     print("NO VALID PIXELS")
        #     print("***ERROR***")
        #     print('error')
         
     
        
        if loop == 0: 
            x, y = choose_roi(img_lat,img_lon,img_i_470)
        
        #3x3 ROI for mean
        x_b = x-25;
        x_a = x+25;
        y_b = y-25;
        y_a = y+25;
# Extract the values from the ROI
        i_355 = np.nanmean(img_i_355[x_b:x_a,y_b:y_a])
        print(img_i_355[x_b:x_a,y_b:y_a])
        i_380 = np.nanmean(img_i_380[x_b:x_a,y_b:y_a])
        i_445 = np.nanmean(img_i_445[x_b:x_a,y_b:y_a])
        i_470 = np.nanmean(img_i_470[x_b:x_a,y_b:y_a])
        i_555 = np.nanmean(img_i_555[x_b:x_a,y_b:y_a])
        i_660 = np.nanmean(img_i_660[x_b:x_a,y_b:y_a])
        i_865 = np.nanmean(img_i_865[x_b:x_a,y_b:y_a])
           
        qs_470 = img_qs_470[x,y]
        qs_660 = img_qs_660[x,y]
        qs_865 = img_qs_865[x,y]
        
        us_470 = img_us_470[x,y]
        us_660 = img_us_660[x,y]
        us_865 = img_us_865[x,y]
        
        qm_470 = img_qm_470[x,y]
        qm_660 = img_qm_660[x,y]
        qm_865 = img_qm_865[x,y]

        um_470 = img_um_470[x,y]
        um_660 = img_um_660[x,y]
        um_865 = img_um_865[x,y]
        
        vaz_355 = img_vaz_355[x,y]
        vaz_380 = img_vaz_380[x,y]
        vaz_445 = img_vaz_445[x,y]
        vaz_470 = img_vaz_470[x,y]
        vaz_555 = img_vaz_555[x,y]
        vaz_660 = img_vaz_660[x,y]
        vaz_865 = img_vaz_865[x,y]
        
        vza_355 = img_vza_355[x,y]    
        vza_380 = img_vza_380[x,y]
        vza_445 = img_vza_445[x,y]
        vza_470 = img_vza_470[x,y]
        vza_555 = img_vza_555[x,y]
        vza_660 = img_vza_660[x,y]
        vza_865 = img_vza_865[x,y]
        
        saz = img_saz[x,y]
        sza = img_sza[x,y]
        
        lat_median = img_lat[x,y]
        lon_median = img_lon[x,y]
        elev_median = img_elev[x,y]
        if(elev_median < 0.0):
            elev_median = 0.0  # Do not allow negative elevation


#________________________Section 2: Geometry Reconciliation___________________________#

        #Input Stokes Parameters
        #470
        stokesin4 = np.array([[qm_470], [um_470]]) #Meridian
        stokesin4s = np.array([[qs_470], [us_470]]) #Scattering
        #660
        stokesin6 = np.array([[qm_660], [um_660]]) #Meridian
        stokesin6s = np.array([[qs_660], [us_660]]) #Scattering
        #865
        stokesin8 = np.array([[qm_865], [um_865]]) #Meridian
        stokesin8s = np.array([[qs_865], [us_865]]) #Scattering
        

        if pol_ref_plane == 'Scattering':
            qg_470, ug_470 = stokesin4s
            qg_660, ug_660 = stokesin6s
            qg_865, ug_865 = stokesin8s
        elif pol_ref_plane == 'Meridian':
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

        if pol_ref_plane == 'Scattering':
            q_median[loop,0] = -eqr_qg_470
            q_median[loop,1] = -eqr_qg_660
            q_median[loop,2] = -eqr_qg_865
    
            u_median[loop,0] = -eqr_ug_470
            u_median[loop,1] = -eqr_ug_660
            u_median[loop,2] = -eqr_ug_865
            
        elif pol_ref_plane == 'Meridian':
            q_median[loop,0] = eqr_qg_470
            q_median[loop,1] = eqr_qg_660
            q_median[loop,2] = eqr_qg_865
    
            u_median[loop,0] = eqr_ug_470
            u_median[loop,1] = eqr_ug_660
            u_median[loop,2] = eqr_ug_865
        

        sza_median[loop] = sza


# #__________________Section 3: Output Data in GRASP SDATA Format__________________#
# Guide to output file names
# NOTE: The options more or less correspond to GRASP retrieval.regime_of_measurement_fitting
#       0 = .radiance (option 1)
#       1-5 = .polarization (option as given)

    print(date_str)
# Change to the output directory
    os.chdir(outpath) 
    
# Generate the base output file name
    #outfile_base = "AirMSPI_"+this_date_str+"_"+this_time_str+"_"
    #outfile_base = outfile_base+this_target_str+"_"
    outfile_base = 'test-Rotfrom'+pol_ref_plane

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

    sdat_date = date_str[0][0:4]+'-'+date_str[0][4:6]+'-'+date_str[0][6:8]
    print(sdat_date)
        
# Parse the time string into the correct format
    print(time_str)
    sdat_time = time_str[0][0:2]+':'+time_str[0][2:4]+':'+time_str[0][4:7]
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
