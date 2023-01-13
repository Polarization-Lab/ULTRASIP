# -*- coding: utf-8 -*-
"""
Geometry Reconciliation 
"""

import numpy as np
import math
import glob
import h5py
import matplotlib.pyplot as plt
import os
import time

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

#%% Data Loading 
datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001208Z_AZ-Prescott_467F_F01_V006.hdf"

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
    
# Set the number of wavelengths for radiometric and polarization separately
#num_int = total number of radiometric channels
#num_pol = total number of polarimetric channels
    
num_step = 1
num_int = 1 
num_pol = 1
    
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

#_________________________Read the data_________________________#
# Set the datasets and read (355 nm)
# Radiometric Channel

# Open the HDF-5 file

inputName = datapath


f = h5py.File(inputName,'r')

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


#_____________________Perform Data Extraction___________________#
# Extract the data in the large bounding box
# NOTE: This puts the array into *image* space       

img_i_470 = np.flipud(I_470[min_y:max_y,min_x:max_x])
img_scat_470 = np.flipud(scat_470[min_y:max_y,min_x:max_x])      
img_vaz_470 = np.flipud(vaz_470[min_y:max_y,min_x:max_x])     
img_vza_470 = np.flipud(vza_470[min_y:max_y,min_x:max_x])
       
        
img_qs_470 = np.flipud(Qs_470[min_y:max_y,min_x:max_x])
img_us_470 = np.flipud(Us_470[min_y:max_y,min_x:max_x])     
img_qm_470 = np.flipud(Qm_470[min_y:max_y,min_x:max_x])
img_um_470 = np.flipud(Um_470[min_y:max_y,min_x:max_x])  
img_ipol_470 = np.flipud(IPOL_470[min_y:max_y,min_x:max_x])      
img_dolp_470 = np.flipud(DOLP_470[min_y:max_y,min_x:max_x])
       
img_saz = np.flipud(saz_470[min_y:max_y,min_x:max_x])
img_sza = np.flipud(sza_470[min_y:max_y,min_x:max_x])
        

        
# Extract the values from the ROI
# NOTE: The coordinates are relative to the flipped "img" array

box_i_470 = img_i_470[box_x1:box_x2,box_y1:box_y2] 
box_scat_470 = img_scat_470[box_x1:box_x2,box_y1:box_y2]     
box_vaz_470 = img_vaz_470[box_x1:box_x2,box_y1:box_y2]       
box_vza_470 = img_vza_470[box_x1:box_x2,box_y1:box_y2]       

box_qs_470 = img_qs_470[box_x1:box_x2,box_y1:box_y2]
box_us_470 = img_us_470[box_x1:box_x2,box_y1:box_y2]

box_qm_470 = img_qm_470[box_x1:box_x2,box_y1:box_y2]
box_um_470 = img_um_470[box_x1:box_x2,box_y1:box_y2]

box_ipol_470 = img_ipol_470[box_x1:box_x2,box_y1:box_y2]
box_dolp_470 = img_dolp_470[box_x1:box_x2,box_y1:box_y2]
       
box_saz = img_saz[box_x1:box_x2,box_y1:box_y2]
box_sza = img_sza[box_x1:box_x2,box_y1:box_y2]
        
        
# Calculate the median

good = (box_i_470 > 0.0)

box_good = box_i_470[good]

i_470 = np.median(box_i_470[good])

scat_470 = np.radians(np.median(box_scat_470[good])) 
vaz_470 = np.radians(np.median(box_vaz_470[good]))
vza_470 = np.radians(np.median(box_vza_470[good]))
  
qs_470 = np.median(box_qs_470[good])
us_470 = np.median(box_us_470[good])  

qm_470 = np.median(box_qm_470[good])
um_470 = np.median(box_um_470[good]) 
    
ipol_470 = np.median(box_ipol_470[good])     
dolp_470 = np.median(box_dolp_470[good])

saz = np.radians(np.median(box_saz[good]))
sza = np.radians(np.median(box_sza[good]))
        
#------------------------------------------Geometry-------------------------
#%%
zenith = np.array([0, 0, 1]);
north = np.array([0, 1, 0]);
view = np.array([np.cos(vaz_470)*np.sin(vza_470), -np.sin(vaz_470)*np.sin(vza_470),-np.cos(vza_470)]);
sun = np.array([np.cos(saz)*np.sin(sza),-np.sin(saz)*np.sin(sza),-np.cos(sza)]);

#print(view)

normal_grasp =  np.cross(zenith,north)/magnitude(np.cross(zenith,north))
vertical_grasp = np.cross(normal_grasp, view)/magnitude(np.cross(normal_grasp, view))
horizontal_grasp = np.cross(vertical_grasp, view)/magnitude(np.cross(vertical_grasp, view))

#AirMSPI Meridian Plane to GRASP
normal_airmer =  np.cross(zenith,view)/magnitude(np.cross(zenith,view))
vertical_airmer = np.cross(normal_airmer, view)/magnitude(np.cross(normal_airmer, view))
horizontal_airmer = np.cross(vertical_airmer, view)/magnitude(np.cross(vertical_airmer, view))

input_matrixmer = np.array([horizontal_airmer,vertical_airmer,view])
output_matrixmer = np.array([horizontal_grasp,vertical_grasp,view])

rot_matrixmer = output_matrixmer.dot(input_matrixmer.transpose())
#print(rot_matrixmer)

delta_alphamer = np.arctan2(rot_matrixmer[0,1],rot_matrixmer[0,0])
#print(delta_alphamer)

rotmat1 = np.array([[np.cos(2*delta_alphamer), np.sin(2*delta_alphamer)],[-np.sin(2*delta_alphamer), np.cos(2*delta_alphamer)]])
polm = np.array([[qm_470],[um_470]])

poloutm = rotmat1.dot(polm)/magnitude(rotmat1.dot(polm))
print(poloutm)


#AirMSPI Scatter Plane to GRASP
normal_airscat =  np.cross(sun,view)/magnitude(np.cross(sun,view))
vertical_airscat = np.cross(normal_airscat, view)/magnitude(np.cross(normal_airscat, view))
horizontal_airscat = np.cross(vertical_airscat, view)/magnitude(np.cross(vertical_airscat, view))

input_matrixscat = np.array([horizontal_airscat,vertical_airscat,view])
output_matrixscat = np.array([horizontal_grasp,vertical_grasp,view])

rot_matrixscat = output_matrixscat.dot(input_matrixscat.transpose())
#print(rot_matrixscat)

delta_alphascat = np.arctan2(rot_matrixscat[0,1],rot_matrixscat[0,0])
#print(delta_alphascat)

rotmat = np.array([[np.cos(2*delta_alphascat), np.sin(2*delta_alphascat)],[-np.sin(2*delta_alphascat), np.cos(2*delta_alphascat)]])
pols = np.array([[qs_470],[us_470]])

polouts = rotmat.dot(pols)/magnitude(rotmat.dot(pols))
print(polouts)




