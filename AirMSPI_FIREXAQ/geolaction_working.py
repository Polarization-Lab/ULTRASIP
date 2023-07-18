# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:22:30 2023

@author: Clarissa
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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
        start_row = mid_row - 1048
        end_row = mid_row + 1048
        start_col = mid_col - 1048
        end_col = mid_col + 1048
        
        a = a[start_row:end_row, start_col:end_col]
        
        return a
    
def plot_radiance(latitude, longitude, radiance):
    plt.figure(figsize=(18, 10))
    plt.scatter(longitude, latitude, c=radiance, cmap='jet', s=5)
    plt.colorbar(label='Radiance')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Radiance over Latitude and Longitude')
    # Add more grid lines
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.001))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    
    # Show every third label on the x-axis
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=0.01))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    
    plt.xticks(rotation=45)
    
    plt.show()

# def get_radiance_at_location(latitude, longitude, radiance, target_lat, target_lon):

#     lat_idx,lon_idx = np.where((latitude == target_lat) & (longitude == target_lon))

#     radiance_at_location = radiance[lat_idx, lon_idx]
#     return radiance_at_location

def get_radiance_at_location(latitude, longitude, radiance, target_lat, target_lon):
    lat_idx, lon_idx = np.where((latitude == target_lat) & (longitude == target_lon))

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        print(f"No radiance data found at Lat: {target_lat}, Lon: {target_lon}")
        return None

    radiance_at_location = radiance[lat_idx[0], lon_idx[0]]
    return radiance_at_location



def main():
    # Change directory to the datapath
    datapath = "C:/Users/Clarissa/Documents/AirMSPI/Washington"
    os.chdir(datapath)
    # Open the HDF file
    file = h5py.File('AirMSPI_ER2_GRP_TERRAIN_20190807_202556Z_WA-Inchelium_571F_F01_V006.hdf', 'r')

    # Read the attribute 'I' from the HDF file
    attribute_data = file['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]

    # Read the latitude and longitude data from the HDF file
    latitude_data = file['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/'][:]
    longitude_data = file['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][:]
    
    radiance = image_crop(attribute_data)
    latitude = image_crop(latitude_data)
    longitude = image_crop(longitude_data)
    
    target_latitude = 47.945  # Replace with your desired latitude
    target_longitude = -118.445  # Replace with your desired longitude

    # Step 2: Plot radiance over latitude and longitude
    plot_radiance(latitude, longitude, radiance)

    # Step 3: Get radiance data for a specific latitude and longitude
    radiance_at_location = get_radiance_at_location(latitude, longitude, radiance, target_latitude, target_longitude)
    print(f"Radiance at Lat: {target_latitude}, Lon: {target_longitude} = {radiance_at_location[0]:.2f}")

if __name__ == "__main__":
    main()
