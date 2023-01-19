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

np.set_printoptions(precision=4)
#%% Data Loading 
#datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001208Z_AZ-Prescott_467F_F01_V006.hdf"
#datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001032Z_AZ-Prescott_646F_F01_V006.hdf"
#datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001344Z_AZ-Prescott_000N_F01_V006.hdf"
#datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001521Z_AZ-Prescott_467A_F01_V006.hdf"
datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data/AirMSPI_ER2_GRP_TERRAIN_20190817_001657Z_AZ-Prescott_646A_F01_V006.hdf"


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
AOLPm_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/AOLP_scatter/'][:]
AOLPs_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/AOLP_meridian/'][:]
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
#Sun azimuth angle (saz) sun zenith angle (sza)
phi_i = saz
theta_i = sza
#theta_i = sza

#View azimuth angle (vaz) view zenith angle (vza)
phi_r = vaz_470
theta_r = vza_470

#Vector Definitions  
#r̂i = [cos Φi sin θi, - sin Φi sin θi, - cos θi] 
#r̂r = [cos Φr sin θr, - sin Φr sin θr, - cos θr].
zenith = np.array([0, 0, 1]);
north = np.array([1, 0, 0]);
illumination =  -np.array([np.cos(phi_i)*np.sin(theta_i),-np.sin(phi_i)*np.sin(theta_i),-np.cos(theta_i)]);
k = -np.array([np.cos(phi_r)*np.sin(theta_r), -np.sin(phi_r)*np.sin(theta_r),-np.cos(theta_r)]);

#GRASP Plane
n_o =  np.cross(north,zenith)/np.linalg.norm(np.cross(north,zenith))
#n_o =  np.cross(zenith,north)/np.linalg.norm(np.cross(zenith,north))
v_o = np.cross(k, n_o)/np.linalg.norm(np.cross(k,n_o))
h_o = np.cross(k,v_o)/np.linalg.norm(np.cross(k,v_o))

#AirMSPI Meridian Plane 
n_i_m =  np.cross(zenith,k)/np.linalg.norm(np.cross(zenith,k))
v_i_m = np.cross(k,n_i_m)/np.linalg.norm(np.cross(k,n_i_m))
h_i_m = np.cross(k,v_i_m)/np.linalg.norm(np.cross(k,v_i_m))

Oin_m = np.array([h_i_m,v_i_m,k])
Oout = np.array([h_o,v_o,k])


Rm = Oout.T@Oin_m

delta_alpham = np.arccos((np.trace(Rm)-1)/2)

rotmat1 = np.array([[np.cos(2*delta_alpham), np.sin(2*delta_alpham)],[-np.sin(2*delta_alpham), np.cos(2*delta_alpham)]])
polm = np.array([[qm_470],[um_470]])
AolPm = 0.5*np.arctan(polm[1,0]/polm[0,0])
DolPm = (polm[0,0]**2 + polm[1,0]**2)**(1/2)

poloutm = rotmat1@(polm)
AolPmout = 0.5*np.arctan(poloutm[1,0]/poloutm[0,0])
DolPmout = (poloutm[0,0]**2 + poloutm[1,0]**2)**(1/2)

poloutmnorm = poloutm/DolPmout
print(poloutm)

#AirMSPI Scatter Plane to GRASP

n_i_s =  np.cross(illumination,k)/np.linalg.norm(np.cross(illumination,k))
v_i_s = np.cross(k,n_i_s)/np.linalg.norm(np.cross(k,n_i_s))
h_i_s = np.cross(k,v_i_s)/np.linalg.norm(np.cross(k,v_i_s))

Oin = np.array([h_i_s,v_i_s,k])


R = Oout.T@Oin

delta_alphascat = np.arccos((np.trace(R)-1)/2)

rotmat = np.array([[np.cos(2*delta_alphascat), np.sin(2*delta_alphascat)],[-np.sin(2*delta_alphascat), np.cos(2*delta_alphascat)]])
pols = np.array([[qs_470],[us_470]])

AolPs = 0.5*np.arctan(pols[1,0]/pols[0,0])
DolPs = (pols[0,0]**2 + pols[1,0]**2)**(1/2)

polouts = rotmat@(pols)
AolPsout = 0.5*np.arctan(polouts[1,0]/polouts[0,0])
DolPsout = (polouts[0,0]**2 + polouts[1,0]**2)**(1/2)
poloutsnorm = polouts/DolPsout


print(polouts)
