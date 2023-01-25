# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:53:12 2023

@author: ULTRASIP_1
"""

"""
DataFormatting_AirMSPI_GRASP.py
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

for g in range(1):  # Main code

# Set the overall timer
    all_start_time = time.time()
#___________________Section 1: Data Products______________________#

#_______________Load in Data___________________#
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/

# Set paths to AirMSPI data and where the output SDATA file should be saved 
# NOTE: datapath is the location of the AirMSPI HDF data files
#       outpath is where the output should be stored
#Work Computer
    datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrieval_1_012423"

#Home Computer 
   # datapath = "C:/Users/Clarissa/Desktop/AirMSPI/Prescott/FIREX-AQ_8212019"
   # outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/SDATA_Files"

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

# Set bounds for the image (USER INPUT)

    min_x = 1900
    max_x = 2200
    min_y = 1900
    max_y = 2200
    
# Set bounds for ROI (USER INPUT)
# Note: These coordinates are RELATIVE to the overall bounding box

    box_x1 = 120
    box_x2 = 125
    box_y1 = 105
    box_y2 = 110
    
#_______________Set Data Extraction Bounds___________________#
# Set the number of wavelengths for radiometric and polarization separately
#num_int = total number of radiometric channels
#num_pol = total number of polarimetric channels

    num_int = 8 
    num_pol = 3
    
# Create arrays to store data
# NOTE: Pay attention to the number of wavelengths

# Angle Arrays
# ALL ANGLES IN RADIANS
    scat_median = np.zeros((num_step,num_int))  # Scattering angle
    vza_median = np.zeros((num_step,num_int))  # View zenith angle
    raz_median = np.zeros((num_step,num_int))  # Relative azimuth angle
    sza_median = np.zeros(num_step)  # Solar zenith angle (one per stare)

#Measurement Arrays   
    i_median = np.zeros((num_step,num_int))  # Intensity
    i_in_polar_median = np.zeros((num_step,num_pol))  # I in polarized bands
    q_median = np.zeros((num_step,num_pol))  # Q
    u_median = np.zeros((num_step,num_pol))  # U
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
        print(error)

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

# Set the timer for reading the file

        start_time = time.time()
        
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

# Print the time

        end_time = time.time()
        print("Time to Read AirMSPI data was %g seconds" % (end_time - start_time))

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
            print(error)
        
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
            print(error)
      
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
### GRASP REQUIRES DATA IN THE MERIDIAN PLANE, BUT THE DEFINITION IS SLIGHTLY
### DIFFERENT THAN THE ONE USED BY AirMSPI, SO CALCULATE THE APPROPRIATE 
### MERIDIAN PLANE HERE
        zenith= np.array([0, 0, 1]);
        nor= np.array([1, 0, 0]);
        i = np.array([np.cos(np.radians(saz))*np.sin(np.radians(-sza)), np.sin(np.radians(saz))*np.sin(np.radians(sza)), -np.cos(np.radians(-sza))]); #illumination vec,flip sign of sza
    
        k = np.array([np.cos(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.sin(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.cos(np.radians(vza_470))]);
        #k_660 = np.array([cosd(vaz_660)*sind(vza_660), -sind(vaz_660)*sind(vza_660), -cosd(vza_660)]);
        #k_865 = np.array([cosd(vaz_865)*sind(vza_865), -sind(vaz_865)*sind(vza_865), -cosd(vza_865)]);

        #Define GRASP Plane   
        grasp_n=np.cross(nor,zenith)/np.linalg.norm(np.cross(nor,zenith));
        grasp_s=np.cross(k,grasp_n)/np.linalg.norm(np.cross(k,grasp_n)); #intersection of transverse & reference
        grasp_p=np.cross(k,grasp_s)/np.linalg.norm(np.cross(k,grasp_s));
    
        #Define airmspi Scattering Plane
        air_ms=np.cross(i,k)/np.linalg.norm(np.cross(i,k));
        air_sS=np.cross(k,air_ms)/np.linalg.norm(np.cross(k,air_ms));
        air_pS=np.cross(k,air_sS)/np.linalg.norm(np.cross(k,air_sS));

        alphas = np.arctan2(np.dot(grasp_p,air_sS),np.dot(grasp_p,air_pS));
        #alphas = np.radians(alphas)

        rotmatrix = np.array([[np.cos(2*alphas),np.sin(2*alphas)],[-np.sin(2*alphas),np.cos(2*alphas)]]);
    
        qg_470, ug_470 = np.dot(rotmatrix,np.array([[qs_470], [us_470]]))
        qg_660, ug_660 = np.dot(rotmatrix,np.array([[qs_660], [us_660]]))
        qg_865, ug_865 = np.dot(rotmatrix,np.array([[qs_865], [us_865]]))

        print(alphas,qs_660,us_660,sza,saz,vza_470,vaz_470,qg_660,ug_660)
        
