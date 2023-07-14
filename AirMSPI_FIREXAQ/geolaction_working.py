# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:13:39 2023

@author: Clarissa
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
        start_row = mid_row - 524
        end_row = mid_row + 524
        start_col = mid_col - 524
        end_col = mid_col + 524
        
        a = a[start_row:end_row, start_col:end_col]
        
        return a

#_______________Start the main code____________#
def main():  # Main code
    # Change directory to the datapath
    datapath = "C:/Users/Clarissa/Documents/AirMSPI/Washington"
    os.chdir(datapath)
    # Open the HDF file
    file = h5py.File('AirMSPI_ER2_GRP_TERRAIN_20190807_182404Z_WA-Inchelium_645F_F01_V006.hdf', 'r')

    # Read the attribute 'I' from the HDF file
    attribute_data = file['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]

    # Read the latitude and longitude data from the HDF file
    latitude_data = file['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/'][:]
    longitude_data = file['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][:]
    
    attribute_data = image_crop(attribute_data)
    latitude_data = image_crop(latitude_data)
    longitude_data = image_crop(longitude_data)
    
    # Prompt the user to enter the center latitude and longitude
    center_lat = float(input("Enter the center latitude: "))
    center_lon = float(input("Enter the center longitude: "))

    # Specify the size of the region of interest (ROI)
    roi_size = 50  # Size of the ROI (10 by 10)

    # Calculate the index ranges for the ROI
    lat_start = np.abs(latitude_data - center_lat).argmin() - roi_size // 2
    lat_end = lat_start + roi_size
    lon_start = np.abs(longitude_data - center_lon).argmin() - roi_size // 2
    lon_end = lon_start + roi_size
    
    # Extract the ROI data
    roi_attribute = attribute_data[lat_start:lat_end, lon_start:lon_end]
    roi_latitude = latitude_data[lat_start:lat_end]
    roi_longitude = longitude_data[lon_start:lon_end]
    
    # Close the HDF file
    file.close()

    # Plot the attribute 'I' over geolocation
    plt.figure(figsize=(10, 8))
    plt.scatter(longitude_data, latitude_data, c=attribute_data, cmap='gray')
    plt.colorbar(label='Attribute I')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Attribute I over Geolocation')
    plt.show()
    
    # Plot the attribute 'I' over the ROI
    plt.figure(figsize=(10, 8))
    plt.scatter(roi_longitude, roi_latitude, c=roi_attribute, cmap='jet')
    plt.colorbar(label='Attribute I')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Attribute I over ROI')
    plt.show()
    
    return roi_attribute
    
### END MAIN FUNCTION
if __name__ == '__main__':
    main() 
     
