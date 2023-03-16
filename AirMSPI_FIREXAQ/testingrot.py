# -*- coding: utf-8 -*-
"""
Code to test rotation methodology 
"""
#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#______________!!! Choose Dataset!!!___________________________#
# #46.7 Degrees in direction of flight -works
# qm_470 = 0.015961975;
# um_470 = 0.0007153307;
# qs_470 = -0.015639367;
# us_470 = 0.0035453732;
# sza=65.30477; #%flipped the sign of raw data
# saz=89.74324;
# vza_470=46.64313275387909;
# vaz_470=219.41413034956926;
# #Nadir % work if change sign i and k vector
qm_470 = 0.024076024;
um_470 = -0.009255913;
qs_470 = -0.025675273;
us_470 = 0.00094201486;
sza = 65.647255; #flipped sign from raw data
saz = 89.98054;
vza_470 = 4.174677;
vaz_470 = 172.39378;
#46.7 Degrees oppostie direction of flight. 
# qm_470 = -0.003341124;
# um_470 = 0.029556079;
# qs_470 = -0.029637378;
# us_470 = 0.0017556292;
# sza = 180-65.96801; #flipped sign of raw  in expression
# saz = 90.20225;
# vza_470 = 41.958958;
# vaz_470 = 46.008686;
#DATA FOR PAPER FROM FILE: AirMSPI_ER2_GRP_TERRAIN_20190817_001208Z_AZ-Prescott_646F_F01_V006.hdf
saz = 89.48908;
sza = 64.939064;
vaz_470 = 220.95282;
vza_470 = 64.98406;
qm_470 = 0.02224747;
um_470 = -0.01015127;
qs_470 = -0.0243936;
us_470 =  0.00118935;
scat_ang = 136.14;


#________________________Geometry Reconciliation___________________________#

zenith= np.array([0, 0, 1]);
nor= np.array([1, 0, 0]);
i = np.array([np.cos(np.radians(saz))*np.sin(np.radians(sza)), -np.sin(np.radians(saz))*np.sin(np.radians(sza)), -np.cos(np.radians(sza))]); #illumination vec,flip sign of sza
k_4 = np.array([np.cos(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.sin(np.radians(vaz_470))*np.sin(np.radians(vza_470)), np.cos(np.radians(vza_470))]);

#AirMSPI Scat 470 nm 
#Define AirMSPI Scattering Plane (input coordinate system) for each wavelength channel
n_i4s = np.cross(i,k_4)/np.linalg.norm(np.cross(i,k_4));
h_i4s=np.cross(k_4,n_i4s)/np.linalg.norm(np.cross(k_4,n_i4s)); #intersection of transverse & reference
v_i4s = np.cross(k_4,h_i4s)/np.linalg.norm(np.cross(k_4,h_i4s));
Oin4s = np.array([h_i4s,v_i4s,k_4]);
        
#Define AirMSPI Meridian Plane (input coordinate system) for each wavelength channel
n_i4 = np.cross(zenith,k_4)/np.linalg.norm(np.cross(zenith,k_4));
h_i4 = np.cross(k_4,n_i4)/np.linalg.norm(np.cross(k_4,n_i4)); #intersection of transverse & reference
v_i4 = np.cross(k_4,h_i4)/np.linalg.norm(np.cross(k_4,h_i4));
Oin4 = np.array([h_i4,v_i4,k_4]);#Meridian    

#GRASP Basis
#n_o = np.cross(k_4,zenith)/np.linalg.norm(np.cross(k_4,zenith));
v_o4 = np.cross(k_4,n_i4)/np.linalg.norm(np.cross(k_4,n_i4)) #intersection of transverse & reference
h_o4 = np.cross(k_4,v_o4)/np.linalg.norm(np.cross(k_4,v_o4))
Oout4 = np.array([h_o4,v_o4]); #GRASP 

#470 nm input
stokesin4 = np.array([[qm_470], [um_470]]) #Meridian
stokesin4s = np.array([[qs_470], [us_470]]) #Scattering

#Scattering to Meridian - AirMSPI
R_nalpha4 = Oin4@(Oin4s.T);
alpha4 = np.arctan2(-R_nalpha4[0,1],R_nalpha4[0,0]);  
rotmatrix4 = np.array([[np.cos(2*alpha4),-np.sin(2*alpha4)],[np.sin(2*alpha4),np.cos(2*alpha4)]]); 
qm_470rot, um_470rot = rotmatrix4@stokesin4s


#Meridian to Scat - AirMSPI
R_nalpha4 = Oin4s@(Oin4.T);
alpha4 = np.arctan2(-R_nalpha4[0,1],R_nalpha4[0,0]);  
rotmatrix4 = np.array([[np.cos(2*alpha4),-np.sin(2*alpha4)],[np.sin(2*alpha4),np.cos(2*alpha4)]]); 
qs_470rot, us_470rot = rotmatrix4@stokesin4


#Meridian AirMSPI to GRASP 
R_nalpha4 = Oout4@(Oin4.T);
alpha4 = np.arctan2(R_nalpha4[0,1],R_nalpha4[0,0]); 
print(np.degrees(alpha4)) 
rotmatrix4 = np.array([[np.cos(2*alpha4),np.sin(2*alpha4)],[-np.sin(2*alpha4),np.cos(2*alpha4)]]); 
qg_470rot, ug_470rot = rotmatrix4@stokesin4

#Scat AirMSPI to GRASP
R_nalpha4 = Oout4@(Oin4s.T);
alpha4 = np.arctan2(-R_nalpha4[0,1],R_nalpha4[0,0]);  
print(np.degrees(alpha4)) 
rotmatrix4 = np.array([[np.cos(2*alpha4),-np.sin(2*alpha4)],[np.sin(2*alpha4),np.cos(2*alpha4)]]); 
qg_470rots, ug_470rots = rotmatrix4@stokesin4s

#Relative Azimuth Calculation
ia = np.array([np.cos(saz),-np.sin(saz)])
ka = np.array([np.cos(vaz_470),-np.sin(vaz_470)])
raz_470m=np.arccos(ia@ka.T);  #range 0 to 180
raz_470m=np.degrees(raz_470m)+180;         #inexplicable GRASP offset

raz_470 = (saz - vaz_470)
if (raz_470 < 0.0):
    raz_470 = raz_470 + 180

# if(raz_470 < 0.0):
#     raz_470 = 360.+raz_470
# if(raz_470 > 180.0):
#     raz_470 = 360.-raz_470
# raz_470 = raz_470+180.

scatt_angle = np.degrees(np.arccos(-i@k_4))

