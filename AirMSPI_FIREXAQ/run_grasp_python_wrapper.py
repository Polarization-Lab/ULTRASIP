# -*- coding: utf-8 -*-
"""
Last edit: Fri Jul  8 11:25:39 2022
@author: C.M.DeLeon
This code acts as a python "wrapper" to run an instance of GRASP. 
Using code from : https://github.com/ReedEspinosa/GSFC-GRASP-Python-Interface


"""

#Import libraries
from runGRASP import graspRun, pixel
import numpy as np
import datetime as dt

# Path to the YAML file you want to use for the aerosol and surface definition
fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/AirMSPI_FIREXAQ/Aug16_2019_RetrievalFiles/Aug16_settings.yml'

#Path to grasp bin/kernel files 
binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'


wvStrFrmt =  '($%4.2f\\mu m$)' # Format of Y-axis labels 
sza = 45 # solar zenith angle
wvls = [0.355,0.380,0.470,0.555] # wavelengths in μm
msTyp = [41] # grasp measurements types (I, Q, U) [must be in ascending order]
azmthΑng = np.r_[0:190:10] # azimuth angles to simulate (0,10,...,175)
vza = 180-np.r_[0:75:5] # viewing zenith angles to simulate [in GRASP cord. sys.]


# prep geometry inputs for GRASP
Nvza = len(vza)
Nazimth = len(azmthΑng)
thtv_g = np.tile(vza, len(msTyp)*len(azmthΑng))
phi_g = np.tile(np.concatenate([np.repeat(φ, len(vza)) for φ in azmthΑng]), len(msTyp))
nbvm = len(thtv_g)/len(msTyp)*np.ones(len(msTyp), int)
meas = np.r_[np.repeat(0.1, nbvm[0]), np.repeat(0.01, nbvm[1]), np.repeat(0.01, nbvm[2])] # dummy values

# define the "pixel" we want to simulate
nowPix = pixel(dt.datetime.now(), 1, 1, 0, 0, masl=0, land_prct=100)
for wvl in wvls: nowPix.addMeas(wvl, msTyp, nbvm, sza, thtv_g, phi_g, meas)

# setup instance of graspRun class, add the above pixel, run grasp and read the output to gr.invRslt[0] (dict)
# releaseYAML=True tmeans that the python will adjust the YAML file to make it correspond to Nwvls (if it does not already)
gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True)
gr.addPix(nowPix)
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)