# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 14:51:02 2023

@author: Clarissa

INPUT: AirMSPI .hdf files
OUTPUT:AirMSPI data products

This is a Python 3.9.13 code to read AirMSPI L1B2 data and 
format the data to perform aerosol retrievals using the 
Generalized Retrieval of Atmosphere and Surface Properties

Code Sections: 
Data products
    a. Load in Data
    b. Set ROI 
    c. Sort and Extract Data
    d. Take Medians 

INPUTS: 
    datapath: directory to .hdf files
    num_step: number of step and stare files in sequence
    sequence_num: Set the index of the sequence of step-and-stare files
                    NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.
    num_rad: number of radiometric channels
    num_pol: number of polarimetric channels

OUTPUTS: 

"""
#_______________Import Packages_________________#
import glob
import h5py
import numpy as np
import os
import time

def read_data(datapath,num_step,sequence_num,num_int,num_pol):

    #Array Definitions
    wavelens = np.empty((num_step,num_int))
    i = np.empty((num_step,num_int))      
    view_zen = np.empty((num_step,num_int))
    view_az = np.empty((num_step,num_int))    
    ipol = np.empty((num_step,num_pol))
    qm = np.empty((num_step,num_pol))
    um = np.empty((num_step,num_pol))
    dolp = np.empty((num_step,num_pol))
    
    sun_zen = np.empty((num_step,1))
    sun_az = np.empty((num_step,1))
    E0_values = np.empty((num_step,num_int))

    esd = 0.0  # Earth-Sun distance (only need one)

    #Center point Arrays
    center_wave = np.zeros(num_int)  # Center wavelengths  
    center_pol = np.zeros(num_pol)  # Center wavelengths (polarized only)
    
    # Calculate the middle of the sequence
    mid_step = int(num_step/2)  
    
    # Crop images to same area to correct for parallax and set a region of interest
    # (ROI) to extract the data from
    
    # Set bounds for the image (USER INPUT)
    min_x = 1900
    max_x = 2200
    min_y = 1900
    max_y = 2200
            
    # Set bounds for ROI (USER INPUT)
    # Note: These coordinates are RELATIVE to the overall bounding box
    roi_x1 = 120
    roi_x2 = 125
    roi_y1 = 105
    roi_y2 = 110
    
    # Change directory to the datapath
    os.chdir(datapath)
    
    # Get the list of files in the directory
    # NOTE: Python returns the files in a strange order, so they will need to be sorted by time
    #Search for files with the correct names
    search_str = '*TERRAIN*.hdf'
    file_list = np.array(glob.glob(search_str))
    
    # Get the number of files    
    num_files = len(file_list)
            
    # Check the number of files against the index to only read one measurement sequence
    print("AirMSPI Files Found: ",num_files)
    sequence_files = file_list[sequence_num*5:sequence_num*5+num_step]
    
    for num_step in range(num_step):
        print(num_step)
        inputName = sequence_files[num_step]
        f = h5py.File(inputName,'r')   
            
        channel355 = '/HDFEOS/GRIDS/355nm_band/Data Fields/';
        i_355 = np.median(np.flipud(f[channel355+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vaz_355 = np.median(np.flipud(f[channel355+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_355 = np.median(np.flipud(f[channel355+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
            
        channel380 = '/HDFEOS/GRIDS/380nm_band/Data Fields/';
        i_380 = np.median(np.flipud(f[channel380+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])          
        vaz_380 = np.median(np.flipud(f[channel380+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_380 = np.median(np.flipud(f[channel380+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
            
        channel445 = '/HDFEOS/GRIDS/445nm_band/Data Fields/';
        i_445 = np.median(np.flipud(f[channel445+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        vaz_445 = np.median(np.flipud(f[channel445+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_445 = np.median(np.flipud(f[channel445+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
            
        channel470 = '/HDFEOS/GRIDS/470nm_band/Data Fields/';
        i_470 = np.median(np.flipud(f[channel470+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        ipol_470 = np.median(np.flipud(f[channel470+'IPOL/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        qm_470 = np.median(np.flipud(f[channel470+'Q_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        um_470 = np.median(np.flipud(f[channel470+'U_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        dolp_470 = np.median(np.flipud(f[channel470+'DOLP/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
            
        vaz_470 = np.median(np.flipud(f[channel470+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_470 = np.median(np.flipud(f[channel470+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
        saz_470 = np.median(np.flipud(f[channel470+'Sun_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        sza_470 = np.median(np.flipud(f[channel470+'Sun_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
            
        channel555 = '/HDFEOS/GRIDS/555nm_band/Data Fields/';
        i_555 = np.median(np.flipud(f[channel555+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        vaz_555 = np.median(np.flipud(f[channel555+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_555 = np.median(np.flipud(f[channel555+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
            
        channel660 = '/HDFEOS/GRIDS/660nm_band/Data Fields/';
        i_660 = np.median(np.flipud(f[channel660+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        ipol_660 = np.median(np.flipud(f[channel660+'IPOL/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        qm_660 = np.median(np.flipud(f[channel660+'Q_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        um_660 = np.median(np.flipud(f[channel660+'U_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        dolp_660 = np.median(np.flipud(f[channel660+'DOLP/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
            
        vaz_660 = np.median(np.flipud(f[channel660+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_660 = np.median(np.flipud(f[channel660+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
            
            
        channel865 = '/HDFEOS/GRIDS/865nm_band/Data Fields/';
        i_865 = np.median(np.flipud(f[channel865+'I/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        ipol_865 = np.median(np.flipud(f[channel865+'IPOL/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        qm_865 = np.median(np.flipud(f[channel865+'Q_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        um_865 = np.median(np.flipud(f[channel865+'U_meridian/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
        dolp_865 = np.median(np.flipud(f[channel865+'DOLP/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])
            
        vaz_865 = np.median(np.flipud(f[channel865+'View_azimuth/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])        
        vza_865 = np.median(np.flipud(f[channel865+'View_zenith/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2]) 
            
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
                
            # Get the navigation information if this is the center acquisition
        if(num_step == mid_step): #latitude and longitude chosen from nadir of step and stare
                
            print("GETTING NAVIGATION")
                    
            # Set the datasets and read (Ancillary)
            evel_coord = np.median(np.flipud(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])   
            lat_coord = np.median(np.flipud(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])   
            long_coord = np.median(np.flipud(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][:][min_y:max_y,min_x:max_x])[roi_x1:roi_x2,roi_y1:roi_y2])   

            #____________________________STORE THE DATA____________________________#
        
        
        intensity = np.array([i_355,i_380,i_445,i_470,i_555,i_660,i_865])
        E0s = np.array([E0_355,E0_380,E0_445,E0_470,E0_555,E0_660,E0_865])
        ipols = np.array([ipol_470,ipol_660,ipol_865])
        qms = np.array([qm_470,qm_660,qm_865])
        ums = np.array([um_470,um_660,um_865])
        dolpms = np.array([dolp_470,dolp_660,dolp_865])

        vza = np.array([vza_355,vza_380,vza_445,vza_470,vza_555,vza_660,vza_865])
        vaz = np.array([vaz_355,vaz_380,vaz_445,vaz_470,vaz_555,vaz_660,vaz_865])
        

                
        for num_int in range(num_int):        
            i[num_step,num_int] = intensity[num_int]
            view_zen[num_step,num_int] = vza[num_int] 
            view_az[num_step,num_int] = vaz[num_int] 
            E0_values[num_step,num_int] = E0s[num_int]
        
        for num_pol in range(num_pol):
            ipol[num_step,num_pol] = ipols[num_pol] 
            qm[num_step,num_pol] = qms[num_pol]
            um[num_step,num_pol] = ums[num_pol]
            dolp[num_step,num_pol] = dolpms[num_pol]
                        
        f.close()
    
    return esd,evel_coord,lat_coord,long_coord,i[:],view_zen[:],view_az[:],E0_values[:],ipol[:],qm[:],um[:],dolp[:]

if __name__ == 'read_data':
    

        print('hello')
    #Work Computer
        datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/"
        outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/2_021623"

    #Home Computer 
       # datapath = "C:/Users/Clarissa/Desktop/AirMSPI/Prescott/FIREX-AQ_8212019"
       # outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/SDATA_Files"

    # Load in the set of measurement sequences
    # Set the length of one measurement sequence of step-and-stare observations
    # NOTE: This will typically be an odd number (9,7,5,...)

        num_step = 5
        
        
    # Set the index of the sequence of step-and-stare files
    # NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

        step_ind = 0
        
        
        num_int = 8 
        num_pol = 3
        
        esd,evel_coord,lat_coord,long_coord,i,view_zen,view_az,E0_values,ipol,qm,um,dolp = read_data(datapath,num_step,step_ind,num_int,num_pol) 