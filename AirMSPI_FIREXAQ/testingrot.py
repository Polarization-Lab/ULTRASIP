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
#46.7 Degrees in direction of flight -works
# qm_470 = 0.015961975;
# um_470 = 0.0007153307;
# qs_470 = -0.015639367;
# us_470 = 0.0035453732;
# sza=-65.30477; #%flipped the sign of raw data
# saz=89.74324;
# vza_470=46.64313275387909;
# vaz_470=219.41413034956926;
# #Nadir % work if change sign i and k vector
# qm_470 = 0.024076024;
# um_470 = -0.009255913;
# qs_470 = -0.025675273;
# us_470 = 0.00094201486;
# sza = -65.647255; #flipped sign from raw data
# saz = 89.98054;
# vza = 4.174677;
# vaz = 172.39378;
#46.7 Degrees oppostie direction of flight. 
qm_470 = -0.003341124;
um_470 = 0.029556079;
qs_470 = -0.029637378;
us_470 = 0.0017556292;
sza = 65.96801; #flipped sign of raw data
saz = 90.20225;
vza_470 = 41.958958;
vaz_470 = 46.008686;


#________________________Geometry Reconciliation___________________________#

zenith= np.array([0, 0, 1]);
nor= np.array([1, 0, 0]);
i = np.array([np.cos(np.radians(saz))*np.sin(np.radians(-sza)), np.sin(np.radians(saz))*np.sin(np.radians(sza)), -np.cos(np.radians(-sza))]); #illumination vec,flip sign of sza

k_4 = np.array([np.cos(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.sin(np.radians(vaz_470))*np.sin(np.radians(vza_470)), -np.cos(np.radians(vza_470))]);

#Define GRASP Plane (output coordinate system) for each wavelength channel
n_o = np.cross(nor,zenith)/np.linalg.norm(np.cross(nor,zenith));
        
#GRASP 470 nm 
v_o4 = np.cross(k_4,n_o)/np.linalg.norm(np.cross(k_4,n_o)) #intersection of transverse & reference
h_o4 = np.cross(k_4,v_o4)/np.linalg.norm(np.cross(k_4,v_o4))
        
#Define AirMSPI Scattering Plane (input coordinate system) for each wavelength channel
n_i4s = np.cross(i,k_4)/np.linalg.norm(np.cross(i,k_4));

#AirMSPI Scat 470 nm 
h_i4s=np.cross(k_4,n_i4s)/np.linalg.norm(np.cross(k_4,n_i4s)); #intersection of transverse & reference
v_i4s = np.cross(k_4,h_i4s)/np.linalg.norm(np.cross(k_4,h_i4s));
        
        
#Define AirMSPI Meridian Plane (input coordinate system) for each wavelength channel
n_i4 = np.cross(zenith,k_4)/np.linalg.norm(np.cross(zenith,k_4));

#AirMSPI Meridian 470 nm 
h_i4 = np.cross(k_4,n_i4)/np.linalg.norm(np.cross(k_4,n_i4)); #intersection of transverse & reference
v_i4 = np.cross(k_4,h_i4)/np.linalg.norm(np.cross(k_4,h_i4));
        

#470 nm 
Oout4 = np.array([h_o4,v_o4]); #GRASP
Oin4 = np.array([h_i4,v_i4]);#Meridian
Oin4s = np.array([h_i4s,v_i4s]); #Scattering
stokesin4 = np.array([[qm_470], [um_470]]) #Meridian
stokesin4s = np.array([[qs_470], [us_470]]) #Scattering

# R_nalpha4 = Oout4@Oin4.T;
# alpha4 = np.arctan2(R_nalpha4[0,1],R_nalpha4[0,0]);  
# rotmatrix4 = np.array([[np.cos(2*alpha4),-np.sin(2*alpha4)],[np.sin(2*alpha4),np.cos(2*alpha4)]]); 
# qg_470, ug_470 = rotmatrix4@stokesin4

R_nalpha4s = Oout4@(Oin4s.T);
alpha4s = np.arctan2(R_nalpha4s[0,1],R_nalpha4s[0,0]);  
rotmatrix4s = np.array([[np.cos(2*alpha4s),-np.sin(2*alpha4s)],[np.sin(2*alpha4s),np.cos(2*alpha4s)]]); 
qgs_470, ugs_470 = rotmatrix4s@stokesin4s

# print('alpha m', alpha4)
print('input Q,U m',stokesin4[0],stokesin4[1])
# print('output Q,U m', qg_470,ug_470)

print('alphas', alpha4s)
print('input Q,Us',stokesin4s[0],stokesin4s[1])
print('output Q,Us', qgs_470,ugs_470)