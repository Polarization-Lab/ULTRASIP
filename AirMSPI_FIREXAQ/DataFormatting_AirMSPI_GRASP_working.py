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
    d. Take Medians 
2. Geometry Reconciliation 
    a. Put AirMSPI measurements into GRASP geometry 
    b. Normalize radiances
3. Structure the data products according to the GRASP SDATA format 
    a. First Output File: Radiometry only in all bands except 935nm
    b. Second Output File: I, Q, U in polarized bands only
    c. Third Output File: I in Radiometric Bands and I,Q,U in polarized bands
4. Visualize ROI and Measurement Values
    a. Intensity vs Scattering Angle
    b. Q and U vs Scattering Angle
    c. Ipol vs Scattering Angle
    d. Degree of Linear Polarization (DoLP) vs Scattering Angle
    
More info on the GRASP geometry and sdata format can be found at grasp-open.com
and more info on this algoritm can be found in DeLeon et. al. (YYYY)

Creation Date: 2022-08-05
Last Modified: 2022-12-01

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

#_______________Start the main code____________#
def main():  # Main code

#___________________Section 1: Data Products______________________#

#_______________Load in Data___________________#
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/

# Set paths to AirMSPI data and where the output SDATA file should be saved 
# NOTE: datapath is the location of the AirMSPI HDF data files
#       outpath is where the output should be stored
#Work Computer
    #datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    datapath = "C:/Users/ULTRASIP_1/Documents/Bakersfield707_DataCopy/"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Mar2723/Scat_FIREX"
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Apr1223/1_Bakersfield"
    #outpath = "C:/Users/ULTRASIP_1/Desktop/ForGRASP/Retrieval_Files"


# #Home Computer 
#     datapath = "C:/Users/Clarissa/Documents/AirMSPI/Prescott/FIREX-AQ_8172019"
#     outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Mar1523/1_FIREX"

# Load in the set of measurement sequences
# Set the length of one measurement sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    
# Calculate the middle of the sequence

    mid_step = int(num_step/2)  
    
# Set the index of the sequence of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 0
    
#_______________Region of Interest___________________#
# Crop images to same area to correct for parallax and set a region of interest
# (ROI) to extract the data from

# # Set bounds for ROI (USER INPUT)
# # Note: These coordinates are RELATIVE to the overall bounding box

    # #FIREX
    # min_x = 1900
    # max_x = 2200
    # min_y = 1900
    # max_y = 2200
    
    #Bakersfield
    min_x = 1200
    max_x = 1900
    min_y = 1200
    max_y = 1900
# Set some bounds for the sample box (USER INPUT)
# Note: These coordinates are RELATIVE to the overall bounding box
    #FIREX
    # box_x1 = 120
    # box_x2 = 125
    # box_y1 = 105
    # box_y2 = 110

    # #Bakserfield
    box_x1 = 485
    box_x2 = 490
    box_y1 = 485
    box_y2 = 490
    
#_______________Set Data Extraction Bounds___________________#
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
        
        img_i_355 = np.flipud(I_355[min_y:max_y,min_x:max_x])
        img_i_380 = np.flipud(I_380[min_y:max_y,min_x:max_x])
        img_i_445 = np.flipud(I_445[min_y:max_y,min_x:max_x])
        img_i_470 = np.flipud(I_470[min_y:max_y,min_x:max_x])
        img_i_555 = np.flipud(I_555[min_y:max_y,min_x:max_x])
        img_i_660 = np.flipud(I_660[min_y:max_y,min_x:max_x])
        img_i_865 = np.flipud(I_865[min_y:max_y,min_x:max_x])
        
        img_scat_355 = np.flipud(scat_355[min_y:max_y,min_x:max_x])
        img_scat_380 = np.flipud(scat_380[min_y:max_y,min_x:max_x])
        img_scat_445 = np.flipud(scat_445[min_y:max_y,min_x:max_x])
        img_scat_470 = np.flipud(scat_470[min_y:max_y,min_x:max_x])
        img_scat_555 = np.flipud(scat_555[min_y:max_y,min_x:max_x])
        img_scat_660 = np.flipud(scat_660[min_y:max_y,min_x:max_x])
        img_scat_865 = np.flipud(scat_865[min_y:max_y,min_x:max_x])
        
        img_vaz_355 = np.flipud(vaz_355[min_y:max_y,min_x:max_x])
        img_vaz_380 = np.flipud(vaz_380[min_y:max_y,min_x:max_x])
        img_vaz_445 = np.flipud(vaz_445[min_y:max_y,min_x:max_x])
        img_vaz_470 = np.flipud(vaz_470[min_y:max_y,min_x:max_x])
        img_vaz_555 = np.flipud(vaz_555[min_y:max_y,min_x:max_x])
        img_vaz_660 = np.flipud(vaz_660[min_y:max_y,min_x:max_x])
        img_vaz_865 = np.flipud(vaz_865[min_y:max_y,min_x:max_x])
        
        img_vza_355 = np.flipud(vza_355[min_y:max_y,min_x:max_x])
        img_vza_380 = np.flipud(vza_380[min_y:max_y,min_x:max_x])
        img_vza_445 = np.flipud(vza_445[min_y:max_y,min_x:max_x])
        img_vza_470 = np.flipud(vza_470[min_y:max_y,min_x:max_x])
        img_vza_555 = np.flipud(vza_555[min_y:max_y,min_x:max_x])
        img_vza_660 = np.flipud(vza_660[min_y:max_y,min_x:max_x])
        img_vza_865 = np.flipud(vza_865[min_y:max_y,min_x:max_x])        
        
        img_qs_470 = np.flipud(Qs_470[min_y:max_y,min_x:max_x])
        img_qs_660 = np.flipud(Qs_660[min_y:max_y,min_x:max_x])
        img_qs_865 = np.flipud(Qs_865[min_y:max_y,min_x:max_x])
        
        img_us_470 = np.flipud(Us_470[min_y:max_y,min_x:max_x])
        img_us_660 = np.flipud(Us_660[min_y:max_y,min_x:max_x])
        img_us_865 = np.flipud(Us_865[min_y:max_y,min_x:max_x])
        
        img_qm_470 = np.flipud(Qm_470[min_y:max_y,min_x:max_x])
        img_qm_660 = np.flipud(Qm_660[min_y:max_y,min_x:max_x])
        img_qm_865 = np.flipud(Qm_865[min_y:max_y,min_x:max_x])
        
        img_um_470 = np.flipud(Um_470[min_y:max_y,min_x:max_x])
        img_um_660 = np.flipud(Um_660[min_y:max_y,min_x:max_x])
        img_um_865 = np.flipud(Um_865[min_y:max_y,min_x:max_x])
        
        img_ipol_470 = np.flipud(IPOL_470[min_y:max_y,min_x:max_x])
        img_ipol_660 = np.flipud(IPOL_660[min_y:max_y,min_x:max_x])
        img_ipol_865 = np.flipud(IPOL_865[min_y:max_y,min_x:max_x])
        
        img_dolp_470 = np.flipud(DOLP_470[min_y:max_y,min_x:max_x])
        img_dolp_660 = np.flipud(DOLP_660[min_y:max_y,min_x:max_x])
        img_dolp_865 = np.flipud(DOLP_865[min_y:max_y,min_x:max_x])
        
        img_saz = np.flipud(saz_470[min_y:max_y,min_x:max_x])
        img_sza = np.flipud(sza_470[min_y:max_y,min_x:max_x])
        
# If this is the center acquisition, process the navigation information

        if(loop == mid_step):
            img_lat = np.flipud(lat[min_y:max_y,min_x:max_x])
            img_lon = np.flipud(lon[min_y:max_y,min_x:max_x])
            img_elev = np.flipud(elev[min_y:max_y,min_x:max_x])
        
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
        
        box_ipol_470 = img_ipol_470[box_x1:box_x2,box_y1:box_y2]
        box_ipol_660 = img_ipol_660[box_x1:box_x2,box_y1:box_y2]
        box_ipol_865 = img_ipol_865[box_x1:box_x2,box_y1:box_y2]
        
        box_dolp_470 = img_dolp_470[box_x1:box_x2,box_y1:box_y2]
        box_dolp_660 = img_dolp_660[box_x1:box_x2,box_y1:box_y2]
        box_dolp_865 = img_dolp_865[box_x1:box_x2,box_y1:box_y2]
        
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
        
        if(len(box_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print('error')
      
        i_355 = np.median(box_i_355[good])
        i_380 = np.median(box_i_380[good])
        i_445 = np.median(box_i_445[good])
        i_470 = np.median(box_i_470[good])
        i_555 = np.median(box_i_555[good])
        i_660 = np.median(box_i_660[good])
        i_865 = np.median(box_i_865[good])
        
        scat_355 = np.median(box_scat_355[good])
        scat_380 = np.median(box_scat_380[good])
        scat_445 = np.median(box_scat_445[good])
        scat_470 = np.median(box_scat_470[good])
        scat_555 = np.median(box_scat_555[good])
        scat_660 = np.median(box_scat_660[good])
        scat_865 = np.median(box_scat_865[good])
        
        vaz_355 = np.median(box_vaz_355[good])
        vaz_380 = np.median(box_vaz_380[good])
        vaz_445 = np.median(box_vaz_445[good])
        vaz_470 = np.median(box_vaz_470[good])
        vaz_555 = np.median(box_vaz_555[good])
        vaz_660 = np.median(box_vaz_660[good])
        vaz_865 = np.median(box_vaz_865[good])
        
        vza_355 = np.median(box_vza_355[good])
        vza_380 = np.median(box_vza_380[good])
        vza_445 = np.median(box_vza_445[good])
        vza_470 = np.median(box_vza_470[good])
        vza_555 = np.median(box_vza_555[good])
        vza_660 = np.median(box_vza_660[good])
        vza_865 = np.median(box_vza_865[good])
        
        qs_470 = np.median(box_qs_470[good])
        qs_660 = np.median(box_qs_660[good])
        qs_865 = np.median(box_qs_865[good])
        
        us_470 = np.median(box_us_470[good])
        us_660 = np.median(box_us_660[good])
        us_865 = np.median(box_us_865[good])
        
        qm_470 = np.median(box_qm_470[good])
        qm_660 = np.median(box_qm_660[good])
        qm_865 = np.median(box_qm_865[good])
        
        um_470 = np.median(box_um_470[good])
        um_660 = np.median(box_um_660[good])
        um_865 = np.median(box_um_865[good])
        
        ipol_470 = np.median(box_ipol_470[good])
        ipol_660 = np.median(box_ipol_660[good])
        ipol_865 = np.median(box_ipol_865[good])
        
        dolp_470 = np.median(box_dolp_470[good])
        dolp_660 = np.median(box_dolp_660[good])
        dolp_865 = np.median(box_dolp_865[good])
        
        saz = np.median(box_saz[good])
        sza = np.median(box_sza[good])
        
        
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
        
        #Vector Definitions
        zenith= np.array([0, 0, 1]);

        i = np.array([np.cos(np.radians(saz))*np.sin(np.radians(sza)), -np.sin(np.radians(saz))*np.sin(np.radians(sza)), -np.cos(np.radians(sza))]); #illumination vec,flip sign of sza
        k_4 = np.array([np.cos(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.sin(np.radians(vaz_470))*np.sin(np.radians(vza_470)), np.cos(np.radians(vza_470))]);
        k_6 = np.array([np.cos(np.radians(vaz_660))*np.sin(np.radians(vza_660)), -np.sin(np.radians(vaz_660))*np.sin(np.radians(vza_660)), np.cos(np.radians(vza_660))]);
        k_8 = np.array([np.cos(np.radians(vaz_865))*np.sin(np.radians(vza_865)), -np.sin(np.radians(vaz_865))*np.sin(np.radians(vza_865)), np.cos(np.radians(vza_865))]);

        # #Define AirMSPI Scattering Plane (input coordinate system) for each wavelength channel
        # #AirMSPI Scat 470 nm
        # n_i4s = np.cross(i,k_4)/np.linalg.norm(np.cross(i,k_4));
        # h_i4s=np.cross(k_4,n_i4s)/np.linalg.norm(np.cross(k_4,n_i4s)); #intersection of transverse & reference
        # v_i4s = np.cross(k_4,h_i4s)/np.linalg.norm(np.cross(k_4,h_i4s));
        # Oin4s = np.array([h_i4s,v_i4s,k_4]);
        
        # #AirMSPI Scat 660 nm
        # n_i6s = np.cross(i,k_6)/np.linalg.norm(np.cross(i,k_6));
        # h_i6s=np.cross(k_6,n_i6s)/np.linalg.norm(np.cross(k_6,n_i6s)); #intersection of transverse & reference
        # v_i6s = np.cross(k_6,h_i6s)/np.linalg.norm(np.cross(k_6,h_i6s));
        # Oin6s = np.array([h_i6s,v_i6s,k_6]);
        
        # #AirMSPI Scat 865 nm
        # n_i8s = np.cross(i,k_8)/np.linalg.norm(np.cross(i,k_8));
        # h_i8s=  np.cross(k_8,n_i8s)/np.linalg.norm(np.cross(k_8,n_i8s)); #intersection of transverse & reference
        # v_i8s = np.cross(k_8,h_i8s)/np.linalg.norm(np.cross(k_8,h_i8s));
        # Oin8s = np.array([h_i8s,v_i8s,k_8]);
        
        #Define AirMSPI Meridian Plane (input coordinate system) for each wavelength channel
        #AirMSPI Mer 470 nm
        n_i4 = np.cross(zenith,k_4)/np.linalg.norm(np.cross(zenith,k_4));
        h_i4 = np.cross(k_4,n_i4)/np.linalg.norm(np.cross(k_4,n_i4)); #intersection of transverse & reference
        v_i4 = np.cross(k_4,h_i4)/np.linalg.norm(np.cross(k_4,h_i4));
        Oin4 = np.array([h_i4,v_i4,k_4]);
        
        #AirMSPI Mer 660 nm
        n_i6 = np.cross(zenith,k_6)/np.linalg.norm(np.cross(zenith,k_6));
        h_i6 = np.cross(k_6,n_i6)/np.linalg.norm(np.cross(k_6,n_i6)); #intersection of transverse & reference
        v_i6 = np.cross(k_6,h_i6)/np.linalg.norm(np.cross(k_6,h_i6));
        Oin6 = np.array([h_i6,v_i6,k_6]);
        
        #AirMSPI Mer 865 nm
        n_i8 = np.cross(zenith,k_8)/np.linalg.norm(np.cross(zenith,k_8));
        h_i8 = np.cross(k_8,n_i8)/np.linalg.norm(np.cross(k_8,n_i8)); #intersection of transverse & reference
        v_i8 = np.cross(k_8,h_i8)/np.linalg.norm(np.cross(k_8,h_i8));
        Oin8 = np.array([h_i8,v_i8,k_8]);
        

        #GRASP Meridian Basis
        n_o4 = n_i4;
        n_o6 = n_i6;
        n_o8 = n_i8;
        
        #470nm
        v_o4 = np.cross(k_4,n_o4)/np.linalg.norm(np.cross(k_4,n_o4)) #intersection of transverse & reference
        h_o4 = np.cross(k_4,v_o4)/np.linalg.norm(np.cross(k_4,v_o4))
        Oout4 = np.array([h_o4,v_o4,k_4]); #GRASP 
        
        #660nm
        v_o6 = np.cross(k_6,n_o6)/np.linalg.norm(np.cross(k_6,n_o6)) #intersection of transverse & reference
        h_o6 = np.cross(k_6,v_o6)/np.linalg.norm(np.cross(k_6,v_o6))
        Oout6 = np.array([h_o6,v_o6,k_6]); #GRASP 
        
        #865nm
        v_o8 = np.cross(k_8,n_o8)/np.linalg.norm(np.cross(k_8,n_o8)) #intersection of transverse & reference
        h_o8 = np.cross(k_8,v_o8)/np.linalg.norm(np.cross(k_8,v_o8))
        Oout8 = np.array([h_o8,v_o8,k_8]); #GRASP 
        
        # #GRASP Scattering Basis
        # n_o4 = n_i4s #np.cross(nor,zenith)/np.linalg.norm(np.cross(nor,zenith));
        # n_o6 = n_i6s
        # n_o8 = n_i8s
        
        # #470nm
        # v_o4 = np.cross(k_4,n_o4)/np.linalg.norm(np.cross(k_4,n_o4)) #intersection of transverse & reference
        # h_o4 = np.cross(k_4,v_o4)/np.linalg.norm(np.cross(k_4,v_o4))
        # Oout4 = np.array([h_o4,v_o4,k_4]); #GRASP 
        
        # #660nm
        # v_o6 = np.cross(k_6,n_o6)/np.linalg.norm(np.cross(k_6,n_o6)) #intersection of transverse & reference
        # h_o6 = np.cross(k_6,v_o6)/np.linalg.norm(np.cross(k_6,v_o6))
        # Oout6 = np.array([h_o6,v_o6,k_6]); #GRASP 
        
        # #865nm
        # v_o8 = np.cross(k_8,n_o8)/np.linalg.norm(np.cross(k_8,n_o8)) #intersection of transverse & reference
        # h_o8 = np.cross(k_8,v_o8)/np.linalg.norm(np.cross(k_8,v_o8))
        # Oout8 = np.array([h_o8,v_o8,k_8]); #GRASP 

    
        # # #Meridian AirMSPI to GRASP 
        R_nalpha4 = Oout4@(Oin4.T);
        alpha4 = np.arctan2(-R_nalpha4[0,1],R_nalpha4[0,0]);  
        rotmatrix4 = np.array([[np.cos(2*alpha4),np.sin(2*alpha4)],[-np.sin(2*alpha4),np.cos(2*alpha4)]]); 
        qg_470, ug_470 = rotmatrix4@stokesin4
        
        R_nalpha6 = Oout6@(Oin6.T);
        alpha6 = np.arctan2(-R_nalpha6[0,1],R_nalpha6[0,0]);  
        rotmatrix6 = np.array([[np.cos(2*alpha6),np.sin(2*alpha6)],[-np.sin(2*alpha6),np.cos(2*alpha6)]]); 
        qg_660, ug_660 = rotmatrix6@stokesin6
        
        R_nalpha8 = Oout8@(Oin8.T);
        alpha8 = np.arctan2(-R_nalpha8[0,1],R_nalpha8[0,0]);  
        rotmatrix8 = np.array([[np.cos(2*alpha8),np.sin(2*alpha8)],[-np.sin(2*alpha8),np.cos(2*alpha8)]]); 
        qg_865, ug_865 = rotmatrix8@stokesin8
        
        # # print("angles:",R_nalpha8,R_nalpha6,R_nalpha4)
        
        # #Scat AirMSPI to GRASP
        # #470
        # R_nalpha4 = Oout4@(Oin4s.T);
        # alpha4 = np.arctan2(-R_nalpha4[0,1],R_nalpha4[0,0]);  
        # rotmatrix4 = np.array([[np.cos(2*alpha4),-np.sin(2*alpha4)],[np.sin(2*alpha4),np.cos(2*alpha4)]]); 
        # qg_470, ug_470 = -stokesin4s #rotmatrix4@stokesin4s
        
        # #660
        # R_nalpha6 = Oout6@(Oin6s.T);
        # alpha6 = np.arctan2(-R_nalpha6[0,1],R_nalpha6[0,0]);  
        # rotmatrix6 = np.array([[np.cos(2*alpha6),-np.sin(2*alpha6)],[np.sin(2*alpha6),np.cos(2*alpha6)]]); 
        # qg_660, ug_660 = -stokesin6s #rotmatrix6@stokesin6s
        
        # #865
        # R_nalpha8 = Oout8@(Oin8s.T);
        # alpha8 = np.arctan2(-R_nalpha8[0,1],R_nalpha8[0,0]);  
        # rotmatrix8 = np.array([[np.cos(2*alpha8),-np.sin(2*alpha8)],[np.sin(2*alpha8),np.cos(2*alpha8)]]); 
        # qg_865, ug_865 = -stokesin8s #rotmatrix8@stokesin8s
        
        qg_470, ug_470 = stokesin4
        qg_660, ug_660 = stokesin6
        qg_865, ug_865 = stokesin8
    
        
# Calculate the relative azimuth angle in the GRASP convention
        
        # # # # # if(loop == mid_step):
        # raz_355 = (vaz_355-saz);
        # raz_380 = (vaz_380-saz);
        # raz_445 = (vaz_445-saz);
        # raz_470 = (vaz_470-saz);
        # raz_555 = (vaz_555-saz);
        # raz_660 = (vaz_660-saz);
        # raz_865 = (vaz_865-saz);
        raz_355 = (saz - vaz_355);
        raz_380 = (saz - vaz_380);
        raz_445 = (saz - vaz_445);
        raz_470 = (saz - vaz_470);
        raz_555 = (saz - vaz_555);
        raz_660 = (saz - vaz_660);
        raz_865 = (saz - vaz_865);
        
        if raz_355 < 0.0:
            raz_355 = raz_355 + 360
        if raz_380 < 0.0:
            raz_380 = raz_380 + 360
        if raz_445 < 0.0:
            raz_445 = raz_445 + 360
        if raz_470 < 0.0:
            raz_470 = raz_470 + 360
        if raz_555 < 0.0:
            raz_555 = raz_555 + 360
        if raz_660 < 0.0:
            raz_660 = raz_660 + 360
        if raz_865 < 0.0:
            raz_865 = raz_865 + 360

        # raz_355 = (saz - vaz_355);
        # raz_380 = (saz - vaz_380);
        # raz_445 = (saz - vaz_445);
        # raz_470 = (saz - vaz_470);
        # raz_555 = (saz - vaz_555);
        # raz_660 = (saz - vaz_660);
        # raz_865 = (saz - vaz_865);
        
        # raz_355 = (vaz_355-saz);
        # raz_380 = (vaz_380-saz);
        # raz_445 = (vaz_445-saz);
        # raz_470 = (vaz_470-saz);
        # raz_555 = (vaz_555-saz);
        # raz_660 = (vaz_660-saz);
        # raz_865 = (vaz_865-saz);

        


        print(scat_355,scat_380,scat_445,scat_470,scat_555,scat_660,scat_865)
            
        

        


        
      
        
### NORMALIZE THE RADIANCES TO THE MEAN EARTH-SUN DISTANCE AND CONVERT TO 
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
        
        eqr_ipol_470 = np.pi*ipol_470*esd**2/E0_470
        eqr_ipol_660 = np.pi*ipol_660*esd**2/E0_660
        eqr_ipol_865 = np.pi*ipol_865*esd**2/E0_865

#____________________________STORE THE DATA____________________________#

        i_median[loop,0] = eqr_i_355
        i_median[loop,1] = eqr_i_380
        i_median[loop,2] = eqr_i_445
        i_median[loop,3] = eqr_i_470
        i_median[loop,4] = eqr_i_555
        i_median[loop,5] = eqr_i_660
        i_median[loop,6] = eqr_i_865
                
        scat_median[loop,0] = (scat_355)
        scat_median[loop,1] = (scat_380)
        scat_median[loop,2] = (scat_445)
        scat_median[loop,3] = (scat_470)
        scat_median[loop,4] = (scat_555)
        scat_median[loop,5] = (scat_660)
        scat_median[loop,6] = (scat_865)
        
        
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
        
        i_in_polar_median[loop,0] = eqr_i_470
        i_in_polar_median[loop,1] = eqr_i_660
        i_in_polar_median[loop,2] = eqr_i_865
        
        q_median[loop,0] = eqr_qg_470
        q_median[loop,1] = eqr_qg_660
        q_median[loop,2] = eqr_qg_865
        
        u_median[loop,0] = -eqr_ug_470
        u_median[loop,1] = -eqr_ug_660
        u_median[loop,2] = -eqr_ug_865
          
        ipol_median[loop,0] = eqr_ipol_470
        ipol_median[loop,1] = eqr_ipol_660
        ipol_median[loop,2] = eqr_ipol_865
        
        dolp_median[loop,0] = dolp_470
        dolp_median[loop,1] = dolp_660
        dolp_median[loop,2] = dolp_865
        
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
    outfile_base = 'RotfromMerd'

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

    sza_mean = np.mean(sza_median)

    for loop in range(num_intensity):
        out_str = out_str+'{:16.8f}'.format(sza_mean)
    
# View zenith angle per measurement per wavelength
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
                   
# Endline
       
    out_str = out_str+'\n'

# Write out the line
     
    outputFile.write(out_str)

# Close the output file

    outputFile.close()        
        
### END MAIN FUNCTION
if __name__ == '__main__':
     main() 