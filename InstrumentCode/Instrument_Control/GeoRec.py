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


print(qm_470,qs_470,um_470,us_470,saz,sza,vaz_470,vza_470)

zenith = np.array([0, 0, 1]);
north = np.array([0, 1, 0]);

illumination = np.array([np.cos(saz)*np.sin(sza),-np.sin(saz)*np.sin(sza),-np.cos(sza)]);

k = np.array([np.cos(vaz_470)*np.sin(vza_470), -np.sin(vaz_470)*np.sin(vza_470),-np.cos(vza_470)]);


n_o =  np.cross(zenith,north)/magnitude(np.cross(zenith,north))
v_o = np.cross(n_o, k)/magnitude(np.cross(n_o, k))
h_o = np.cross(v_o, k)/magnitude(np.cross(v_o, k))

#AirMSPI Meridian Plane to GRASP
n_i_m =  np.cross(zenith,k)/magnitude(np.cross(zenith,k))
v_i_m = np.cross(k,n_i_m)/magnitude(np.cross(k,n_i_m))
h_i_m = np.cross(v_i_m, k)/magnitude(np.cross(v_i_m, k))

input_matrixm = np.array([h_i_m,v_i_m])
output_matrixm = np.array([h_o,v_o])

#rot_matrixm = output_matrixm.transpose().dot(input_matrixm)

rot_matrixm = output_matrixm.dot(input_matrixm.transpose())

delta_alpham = np.arctan2(rot_matrixm[0,1],rot_matrixm[0,0])


rotmat1 = np.array([[np.cos(2*delta_alpham), np.sin(2*delta_alpham)],[-np.sin(2*delta_alpham), np.cos(2*delta_alpham)]])
polm = np.array([[qm_470],[um_470]])

poloutm = rotmat1.dot(polm)
print(poloutm)


#AirMSPI Scatter Plane to GRASP
n_i_s =  np.cross(illumination,k)/magnitude(np.cross(illumination,k))
v_i_s = np.cross(k,n_i_s)/magnitude(np.cross(k,n_i_s))
h_i_s = np.cross(v_i_s, k)/magnitude(np.cross(v_i_s, k))

input_matrixs= np.array([h_i_s,v_i_s])
output_matrixs = np.array([h_o,v_o])

#rot_matrixs = output_matrixs.transpose().dot(input_matrixs)

rot_matrixs = output_matrixs.dot(input_matrixs.transpose())

delta_alphascat = np.arctan2(rot_matrixs[0,1],rot_matrixs[0,0])

rotmat = np.array([[np.cos(2*delta_alphascat), np.sin(2*delta_alphascat)],[-np.sin(2*delta_alphascat), np.cos(2*delta_alphascat)]])
pols = np.array([[qs_470],[us_470]])

polouts = rotmat.dot(pols)
print(polouts)


#%%
zenith = np.array([0, 0, 1]);
north = np.array([0, 1, 0]);

illumination = np.array([np.cos(saz)*np.sin(sza),-np.sin(saz)*np.sin(sza),-np.cos(sza)]);

k = np.array([np.cos(vaz_470)*np.sin(vza_470), -np.sin(vaz_470)*np.sin(vza_470),-np.cos(vza_470)]);


v_o =  np.cross(zenith,north)/magnitude(np.cross(zenith,north))
h_o = np.cross(k,v_o)/magnitude(np.cross(k,v_o))

#AirMSPI Meridian Plane to GRASP
v_i_m =  np.cross(zenith,k)/magnitude(np.cross(zenith,k))
h_i_m = np.cross(k,v_i_m)/magnitude(np.cross(k,v_i_m))

input_matrixm = np.array([h_i_m,v_i_m])
output_matrixm = np.array([h_o,v_o])

rot_matrixm = output_matrixm.transpose().dot(input_matrixm)

#rot_matrixm = output_matrixm.dot(input_matrixm.transpose())
delta_alpham = np.arctan2(rot_matrixm[0,1],rot_matrixm[0,0])

rotmat1 = np.array([[np.cos(2*delta_alpham), np.sin(2*delta_alpham)],[-np.sin(2*delta_alpham), np.cos(2*delta_alpham)]])
polm = np.array([[qm_470],[um_470]])

poloutm = rotmat1.dot(polm)
print(poloutm)

#AirMSPI Scatter Plane to GRASP
v_i_s =  np.cross(illumination,k)/magnitude(np.cross(illumination,k))
h_i_s = np.cross(k,v_i_s)/magnitude(np.cross(k,v_i_s))

input_matrixs = np.array([h_i_s,v_i_s])
output_matrixs = np.array([h_o,v_o])

rot_matrixs = output_matrixs.transpose().dot(input_matrixs)

#rot_matrixs = output_matrixs.dot(input_matrixs.transpose())
delta_alphas = np.arctan2(rot_matrixs[0,1],rot_matrixs[0,0])

rotmat = np.array([[np.cos(2*delta_alphas), np.sin(2*delta_alphas)],[-np.sin(2*delta_alphas), np.cos(2*delta_alphas)]])
pols = np.array([[qs_470],[us_470]])

polouts = rotmat.dot(pols)
print(polouts)