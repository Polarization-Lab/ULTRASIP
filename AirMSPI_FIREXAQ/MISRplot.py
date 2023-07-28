# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:58:15 2023

@author: ULTRASIP_1
"""

import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file_path.nc' with the actual path to your .nc file
datapath = 'C:/Users/ULTRASIP_1/Documents/SPIE'

os.chdir(datapath)

file_path = 'MISR_AM1_AS_AEROSOL_P045_O104434_F13_0023.nc'

# Open the .nc file in 'read' mode
nc_file = nc.Dataset(file_path, 'r')

AOD = nc_file['4.4_KM_PRODUCTS/Aerosol_Optical_Depth'][:]  # This will read the entire data for the given variable
lat = nc_file['4.4_KM_PRODUCTS/Latitude'][:] 
lon = nc_file['4.4_KM_PRODUCTS/Longitude'][:] 
time = nc_file['4.4_KM_PRODUCTS/Time'][:] 



lat_slice = lat[1316:1325,398:404]
lon_slice = lon[1316:1325,398:404]
aod_slice = AOD[1316:1325,398:404]


# Close the .nc file when you're done reading
nc_file.close()

# Plot AOD over lat/lon using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(lon, lat, c=AOD, cmap='viridis', s=10)
plt.colorbar(label='Aerosol Optical Depth')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Aerosol Optical Depth over Lat/Lon')
plt.grid(True)
plt.show()

# Plot AOD over lat/lon using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(lon_slice, lat_slice, c=aod_slice, cmap='viridis', s=10)
plt.colorbar(label='Aerosol Optical Depth')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Aerosol Optical Depth over Lat/Lon')
plt.grid(True)
plt.show()
