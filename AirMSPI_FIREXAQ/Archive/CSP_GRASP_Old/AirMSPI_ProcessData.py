# -*- coding: utf-8 -*-
"""
AirMSPI_Preprocessing.py

This is a Python 3.9.13 code to read in AirMSPI level 1 data and process it for
use in GRASP (grasp-open.com) aerosol retrievals 

Creation Date: 2022-08-08
Last Modified: 2022-09-12

by Michael J. Garay and Clarissa M. DeLeon
(Michael.J.Garay@jpl.nasa.gov , cdeleon@arizona.edu)
"""

# Import packages

import glob
import h5py
import numpy as np
import os
import time

# Start the main code

def process():  # Main code

###----------------------Variable Definitions---------------------------###
# Set the paths
# NOTE: datapath is the location of the GRASP output files


    datapath = "C:/Users/ULTRASIP_1/Documents/Prescott_Data"
   
# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    mid_step = int(num_step/2)  # Calculate the middle of the sequence
    n=0;
    
# Set the index of the group of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

# Set the number of wavelengths for intensity and polarization separately

    num_int = 8
    num_pol = 3
    
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

### Create arrays to store data
# NOTE: Pay attention to the number of wavelengths
    data = np.array([], dtype=float, ndmin=2)

    i_median = np.zeros((num_step,num_int))  # Intensity
    
    scat_median = np.zeros((num_step,num_int))  # Scattering angle
    vza_median = np.zeros((num_step,num_int))  # View zenith angle
    raz_median = np.zeros((num_step,num_int))  # Relative azimuth angle
    
    i_in_polar_median = np.zeros((num_step,num_pol))  # I in polarized bands
    q_median = np.zeros((num_step,num_pol))  # Q
    u_median = np.zeros((num_step,num_pol))  # U
    ipol_median = np.zeros((num_step,num_pol))  # Ipol
    dolp_median = np.zeros((num_step,num_pol))  # DoLP
        
    sza_median = np.zeros(num_step)  # Solar zenith angle (one per stare)
    
    center_wave = np.zeros(num_int)  # Center wavelengths
    
    center_pol = np.zeros(num_pol)  # Center wavelengths (polarized only)
    
    esd = 0.0  # Earth-Sun distance (only need one)
    
###----------------------Read the AirMSPI data-----------------------------###

# Change directory to the datapath

    os.chdir(datapath)
    
# Get the list of files in the directory
# NOTE: Python returns the files in a strange order, so they will need to be sorted by time

    search_str = '*TERRAIN*.hdf'
    dum_list = glob.glob(search_str)
    raw_list = np.array(dum_list)  # Convert to a numpy array
    
# Get the number of files
    
    num_files = len(raw_list)

# Set the index of the group of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = int((num_files/num_step) - 1);
    
# Check the number of files against the index

    print("AirMSPI Files Found: ",num_files)
    
    # num_need = num_step*step_ind+num_step
    
    # if(num_need > num_files):
    #     print("***ERROR***")
    #     print("Not Enough Files for Group Index")
    #     print(num_need)
    #     print(num_files)
    #     print("Check index of the group at Line 44")
    #     print("***ERROR***")
    #     #print(error)

# Loop through files and sort by time (HHMMSS)
# Also, extract date and target name

    time_raw = np.zeros((num_files),dtype=int)  # Time as an integer
    
    time_str_raw = []  # Time as a string
    date_str_raw = []  # Date as a string
    target_str_raw = []  # Target as a string    

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
    
    while step_ind >= 0 :    
        
            print("Sequence" + "" + str(step_ind));
        ### Loop through the files for one set of step-and-stare acquisitions

            for loop in range(num_step):
            
                this_ind = loop+num_step*step_ind;
                
                
                
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
                
        # READ THE DATA

        # Set the datasets and read (355 nm)
        # Intensity only

                print("355nm")
                dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/']
                I_355 = dset[:]
                dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/Scattering_angle/']
                scat_355 = dset[:]
                dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_azimuth/']
                vaz_355 = dset[:]
                dset = f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_zenith/']
                vza_355 = dset[:]
                
        # Set the datasets and read (380 nm)
        # Intensity only

                print("380nm")
                dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/']
                I_380 = dset[:]
                dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/Scattering_angle/']
                scat_380 = dset[:]
                dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_azimuth/']
                vaz_380 = dset[:]
                dset = f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_zenith/']
                vza_380 = dset[:]
                
        # Set the datasets and read (445 nm)
        # Intensity only

                print("445nm")
                dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/']
                I_445 = dset[:]
                dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/Scattering_angle/']
                scat_445 = dset[:]
                dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_azimuth/']
                vaz_445 = dset[:]
                dset = f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_zenith/']
                vza_445 = dset[:]
                
        # Set the datasets and read (470 nm)
        # Polarized band (INCLUDE SOLAR ANGLES)
        # NOTE: GRASP wants polarization in the meridian plane, but this needs to be
        #       calculated from the AirMSPI values in the scattering plane

                print("470nm")
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/']
                I_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/DOLP/']
                DOLP_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/IPOL/']
                IPOL_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Scattering_angle/']
                scat_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/Sun_azimuth/']
                saz_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/Sun_zenith/']
                sza_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/']
                Qs_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/']
                Us_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_azimuth/']
                vaz_470 = dset[:]
                dset = f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_zenith/']
                vza_470 = dset[:]
                
        # Set the datasets and read (555 nm)
        # Intensity only (INCLUDE SOLAR ANGLES)

                print("555nm")
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/']
                I_555 = dset[:]
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/Scattering_angle/']
                scat_555 = dset[:]
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_azimuth/']
                vaz_555 = dset[:]
                dset = f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_zenith/']
                vza_555 = dset[:]
                
        # Set the datasets and read (660 nm)
        # Polarized band
        # NOTE: GRASP wants polarization in the meridian plane, but this needs to be
        #       calculated from the AirMSPI values in the scattering plane

                print("660nm")
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/']
                I_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/DOLP/']
                DOLP_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/IPOL/']
                IPOL_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Scattering_angle/']
                scat_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_scatter/']
                Qs_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_scatter/']
                Us_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_azimuth/']
                vaz_660 = dset[:]
                dset = f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_zenith/']
                vza_660 = dset[:]
                
        # Set the datasets and read (865 nm)
        # Polarized band
        # NOTE: GRASP wants polarization in the meridian plane, but this needs to be
        #       calculated from the AirMSPI values in the scattering plane

                print("865nm")
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/']
                I_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/DOLP/']
                DOLP_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/IPOL/']
                IPOL_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Scattering_angle/']
                scat_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_scatter/']
                Qs_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_scatter/']
                Us_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_azimuth/']
                vaz_865 = dset[:]
                dset = f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_zenith/']
                vza_865 = dset[:]
                
        # Get the Earth-Sun distance from the file attributes from the first file

                if(esd == 0.0):
                    print("GETTING EARTH-SUN DISTANCE")
                    attr = f['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/'].attrs['Sun distance']
                    esd = attr
                    
        # Get the actual center wavelengths and E0 values

                    dset = f['/Channel_Information/Center_wavelength/']
                    center_raw = dset[:]
                
                    dset = f['/Channel_Information/Solar_irradiance_at_1_AU/']
                    E0_wave = dset[:]
                    
        # Calculate the effective center wavelengths by appropriate averaging
        # NOTE: Essentially, for intensity only bands, the center wavelength is given in the
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
        # NOTE: Essentially, for intensity only bands, the E0 is given in the
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

                if(loop == mid_step):
                    
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
        #       are changed, then the test needs to change    
                
                good = ((img_i_355 > 0.0) & (img_i_380 > 0.0) & (img_i_445 > 0.0) &
                    (img_i_470 > 0.0) & (img_i_555 > 0.0) & (img_i_660 > 0.0) &
                    (img_i_865 > 0.0))
                    
                img_good = img_i_355[good]
                
                if(len(img_good) < 1):
                    print("***ERROR***")
                    print("NO VALID PIXELS")
                    print("***ERROR***")
                    #print(error)
                
        # Extract the values for the interior box
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
                    #print(error)
              
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
                    
        ### GRASP REQUIRES DATA IN THE MERIDIAN PLANE, BUT THE DEFINITION IS SLIGHTLY
        ### DIFFERENT THAN THE ONE USED BY AirMSPI, SO CALCULATE THE APPROPRIATE 
        ### MERIDIAN PLANE HERE

                mu0 = np.cos(np.radians(sza))
                nu0 = np.sin(np.radians(sza))

                mu_470 = np.cos(np.radians(vza_470))
                mu_660 = np.cos(np.radians(vza_660))
                mu_865 = np.cos(np.radians(vza_865))
                
                nu_470 = np.sin(np.radians(vza_470))
                nu_660 = np.sin(np.radians(vza_660))
                nu_865 = np.sin(np.radians(vza_865))
                
                delta_phi_470 = vaz_470 - saz
                delta_phi_660 = vaz_660 - saz
                delta_phi_865 = vaz_865 - saz
                
                x1_470 = nu0*np.sin(np.radians(delta_phi_470))
                x1_660 = nu0*np.sin(np.radians(delta_phi_660))
                x1_865 = nu0*np.sin(np.radians(delta_phi_865))
                
                x2_470 = nu_470*mu0+mu_470*nu0*np.cos(np.radians(delta_phi_470))
                x2_660 = nu_660*mu0+mu_660*nu0*np.cos(np.radians(delta_phi_660))
                x2_865 = nu_865*mu0+mu_865*nu0*np.cos(np.radians(delta_phi_865))
                
                alpha_470 = np.arctan2(x1_470,x2_470) #  Returns value in radians
                alpha_660 = np.arctan2(x1_660,x2_660) #  Returns value in radians
                alpha_865 = np.arctan2(x1_865,x2_865) #  Returns value in radians
                
        # Convert to GRASP coordinates
                
                qg_470 = qs_470*np.cos(2.0*alpha_470)+us_470*np.sin(2.0*alpha_470)
                qg_660 = qs_660*np.cos(2.0*alpha_660)+us_660*np.sin(2.0*alpha_660)
                qg_865 = qs_865*np.cos(2.0*alpha_865)+us_865*np.sin(2.0*alpha_865)

                ug_470 = qs_470*np.sin(2.0*alpha_470)+us_470*np.cos(2.0*alpha_470)
                ug_660 = qs_660*np.sin(2.0*alpha_660)+us_660*np.cos(2.0*alpha_660)
                ug_865 = qs_865*np.sin(2.0*alpha_865)+us_865*np.cos(2.0*alpha_865)

        # Calculate the relative azimuth angle in the GRASP convention
        # NOTE: This bit of code seems kludgy and comes from older AirMSPI code

                raz_355 = saz - vaz_355
                if(raz_355 < 0.0):
                    raz_355 = 360.+raz_355
                if(raz_355 > 180.0):
                    raz_355 = 360.-raz_355
                raz_355 = raz_355+180.
                
                raz_380 = saz - vaz_380
                if(raz_380 < 0.0):
                    raz_380 = 360.+raz_380
                if(raz_380 > 180.0):
                    raz_380 = 360.-raz_380
                raz_380 = raz_380+180.
                
                raz_445 = saz - vaz_445
                if(raz_445 < 0.0):
                    raz_445 = 360.+raz_445
                if(raz_445 > 180.0):
                    raz_445 = 360.-raz_445
                raz_445 = raz_445+180.
                
                raz_470 = saz - vaz_470
                if(raz_470 < 0.0):
                    raz_470 = 360.+raz_470
                if(raz_470 > 180.0):
                    raz_470 = 360.-raz_470
                raz_470 = raz_470+180.
                
                raz_555 = saz - vaz_555
                if(raz_555 < 0.0):
                    raz_555 = 360.+raz_555
                if(raz_555 > 180.0):
                    raz_555 = 360.-raz_555
                raz_555 = raz_555+180.
                
                raz_660 = saz - vaz_660
                if(raz_660 < 0.0):
                    raz_660 = 360.+raz_660
                if(raz_660 > 180.0):
                    raz_660 = 360.-raz_660
                raz_660 = raz_660+180.
                
                raz_865 = saz - vaz_865
                if(raz_865 < 0.0):
                    raz_865 = 360.+raz_865
                if(raz_865 > 180.0):
                    raz_865 = 360.-raz_865
                raz_865 = raz_865+180.
                
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
                
        ### STORE THE DATA

                i_median[loop,0] = eqr_i_355
                i_median[loop,1] = eqr_i_380
                i_median[loop,2] = eqr_i_445
                i_median[loop,3] = eqr_i_470
                i_median[loop,4] = eqr_i_555
                i_median[loop,5] = eqr_i_660
                i_median[loop,6] = eqr_i_865
                
                scat_median[loop,0] = scat_355
                scat_median[loop,1] = scat_380
                scat_median[loop,2] = scat_445
                scat_median[loop,3] = scat_470
                scat_median[loop,4] = scat_555
                scat_median[loop,5] = scat_660
                scat_median[loop,6] = scat_865
                
                vza_median[loop,0] = vza_355
                vza_median[loop,1] = vza_380
                vza_median[loop,2] = vza_445
                vza_median[loop,3] = vza_470
                vza_median[loop,4] = vza_555
                vza_median[loop,5] = vza_660
                vza_median[loop,6] = vza_865
                
                raz_median[loop,0] = raz_355
                raz_median[loop,1] = raz_380
                raz_median[loop,2] = raz_445
                raz_median[loop,3] = raz_470
                raz_median[loop,4] = raz_555
                raz_median[loop,5] = raz_660
                raz_median[loop,6] = raz_865
                
                i_in_polar_median[loop,0] = eqr_i_470
                i_in_polar_median[loop,1] = eqr_i_660
                i_in_polar_median[loop,2] = eqr_i_865
                
                q_median[loop,0] = eqr_qg_470
                q_median[loop,1] = eqr_qg_660
                q_median[loop,2] = eqr_qg_865
                
                u_median[loop,0] = eqr_ug_470
                u_median[loop,1] = eqr_ug_660
                u_median[loop,2] = eqr_ug_865
                
                ipol_median[loop,0] = eqr_ipol_470
                ipol_median[loop,1] = eqr_ipol_660
                ipol_median[loop,2] = eqr_ipol_865
                
                dolp_median[loop,0] = dolp_470
                dolp_median[loop,1] = dolp_660
                dolp_median[loop,2] = dolp_865 
                
                sza_median[loop] = sza
        
            array = [i_median[:], scat_median[:],vza_median[:],raz_median[:],i_in_polar_median[:],q_median[:],u_median[:],ipol_median[:],dolp_median[:],sza_median[:]];
             
            data = np.append(data,array[:])
            
            n=n+1;
            step_ind = step_ind - 1 
            
    return data
###----------------------- END MAIN FUNCTION-------------------------------###


# if __name__ == '__process__':
data = process()

