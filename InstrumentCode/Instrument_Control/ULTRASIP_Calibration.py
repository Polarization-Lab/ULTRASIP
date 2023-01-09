# -*- coding: utf-8 -*-
"""
ULTRASIP Calibration Code
@author: C.M.DeLeon
"""

# Import Libraries
import matplotlib.pyplot as plt
import matplotlib.image as im
import cv2
import numpy as np
import time
import sys
sys.path.append(
    'C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/')
sys.path.append(
    'C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/Rotation_Motor')
import Cam_Cmmand as cam
import elliptec

outdir = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/Calibration_01.05.23/"

# Polarizer rotation connect and angles
controller = elliptec.Controller('COM4')
ro = elliptec.Rotator(controller)
angles = [0,90]#, 45, 90, 135]

# Image Acquisition
nb_frames = 5
exposure = 1e-6

time.sleep(4)

# Home the rotator before usage
ro.home()
# set an offset
#offset = 0
offset = -0.025

# Loop over a list of angles and acquire for each
for angle in angles:
    ro.set_angle(angle+offset)
    angleout = ro.get_angle()
    print(angleout)
    
    filename = angleout
    image = cam.takeimage(nb_frames, exposure)
    
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    
    np.savetxt(outdir + "/"+ str(filename)+".csv", image, delimiter=",")
    im.imsave(outdir + "/"+ str(filename)+".jpeg", image)
    
    print(type(image))
    #cv2.imwrite(str(filename) + ".png", image)
    
    time.sleep(3)

# Home and close everything
ro.home()
ro.close()

