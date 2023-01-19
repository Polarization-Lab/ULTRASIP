# -*- coding: utf-8 -*-
"""
Geometry reconciliation  test code
"""

import numpy as np
import math

np.set_printoptions(precision=4)

#Angles and Q and U from AirMSPI File 
qm_470 = 0.015961975;
um_470 = 0.0007153307;

qs_470 = -0.015639367;
us_470 = 0.0035453732;

#Sun azimuth angle (saz) sun zenith angle (sza)
phi_i = saz = 1.566315;
theta_i = sza = - 1.1397834;

#View azimuth angle (vaz) view zenith angle (vza)
phi_r = vaz = 3.829499;
theta_r = vza = 0.81407624; 

#Vector Definitions  
#r̂i = [cos Φi sin θi, - sin Φi sin θi, - cos θi] 
#r̂r = [cos Φr sin θr, - sin Φr sin θr, - cos θr].
zenith = np.array([0, 0, 1]);
north = np.array([1, 0, 0]);
illumination =  np.array([np.cos(phi_i)*np.sin(theta_i),-np.sin(phi_i)*np.sin(theta_i),-np.cos(theta_i)]);
k = np.array([np.cos(phi_r)*np.sin(theta_r), -np.sin(phi_r)*np.sin(theta_r),-np.cos(theta_r)]);

#GRASP Plane
n_o =  np.cross(north,zenith)/np.linalg.norm(np.cross(north,zenith))
h_o = np.cross(k, n_o)/np.linalg.norm(np.cross(k,n_o))
v_o = np.cross(k,h_o)/np.linalg.norm(np.cross(k,h_o))

#AirMSPI Meridian Plane 
n_i_m =  np.cross(zenith,k)/np.linalg.norm(np.cross(zenith,k))
h_i_m = np.cross(k,n_i_m)/np.linalg.norm(np.cross(k,n_i_m))
v_i_m = np.cross(k,h_i_m)/np.linalg.norm(np.cross(k,h_i_m))

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
h_i_s = np.cross(k,n_i_s)/np.linalg.norm(np.cross(k,n_i_s))
v_i_s = np.cross(k,h_i_s)/np.linalg.norm(np.cross(k,h_i_s))

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

