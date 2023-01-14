# -*- coding: utf-8 -*-
"""
Geometry reconciliation  test code
"""

import numpy as np
import math

#Angles and Q and U from AirMSPI File 
qm_470 = 0.015961975;
um_470 = 0.0007153307;

qs_470 = -0.015639367;
us_470 = 0.0035453732;

#Sun azimuth angle (saz) sun zenith angle (sza)
phi_i = saz = 1.566315;
theta_i = sza = 1.1397834;

#View azimuth angle (vaz) view zenith angle (vza)
phi_r = vaz = 3.829499;
theta_r = vza = 0.81407624; 

#Vector Definitions  
#r̂i = [cos Φi sin θi, - sin Φi sin θi, - cos θi] 
#r̂r = [cos Φr sin θr, - sin Φr sin θr, - cos θr].
zenith = np.array([0, 0, 1]);
north = np.array([1, 0, 0]);
illumination = np.array([np.cos(phi_i)*np.sin(theta_i),-np.sin(phi_i)*np.sin(theta_i),-np.cos(theta_i)]);
k = np.array([np.cos(phi_r)*np.sin(theta_r), -np.sin(phi_r)*np.sin(theta_r),-np.cos(theta_r)]);

#GRASP Plane
n_o =  np.cross(zenith,north)/np.linalg.norm(np.cross(zenith,north))
h_o = np.cross(k, n_o)/np.linalg.norm(np.cross(k,n_o))
v_o = np.cross(k,h_o)/np.linalg.norm(np.cross(k,h_o))

#AirMSPI Meridian Plane 
n_i_m =  np.cross(zenith,k)/np.linalg.norm(np.cross(zenith,k))
h_i_m = np.cross(k,n_i_m)/np.linalg.norm(np.cross(k,n_i_m))
v_i_m = np.cross(k,h_i_m)/np.linalg.norm(np.cross(k,h_i_m))

Oin_m = np.array([h_i_m,v_i_m])
Oout_m = np.array([h_o,v_o])

Oin_mt = Oin_m.T

R_m = Oout_m@Oin_m.T

delta_alpham = np.arctan2(R_m[0,1],R_m[0,0])

rotmat1 = np.array([[np.cos(2*delta_alpham), np.sin(2*delta_alpham)],[-np.sin(2*delta_alpham), np.cos(2*delta_alpham)]])
polm = np.array([[qm_470],[um_470]])

poloutm = rotmat1.dot(polm)
print(poloutm)

#AirMSPI Scatter Plane to GRASP
n_i_s =  np.cross(illumination,k)/np.linalg.norm(np.cross(illumination,k))
h_i_s = np.cross(k,n_i_s)/np.linalg.norm(np.cross(k,n_i_s))
v_i_s = np.cross(k,h_i_s)/np.linalg.norm(np.cross(k,h_i_s))

Oin = np.array([h_i_s,v_i_s])
Oout = np.array([h_o,v_o])

R = Oout.T@Oin

delta_alphascat = np.arctan2(R[0,1],R[0,0])

rotmat = np.array([[np.cos(2*delta_alphascat), np.sin(2*delta_alphascat)],[-np.sin(2*delta_alphascat), np.cos(2*delta_alphascat)]])
pols = np.array([[qs_470],[us_470]])

polouts = rotmat.dot(pols)

print(polouts)

