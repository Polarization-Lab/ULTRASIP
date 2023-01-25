# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:29:45 2023
This code runs ULTRASIP Measurement
@author: C.M.DeLeon
"""
# Import Libraries
import matplotlib.pyplot as plt
import serial
import time
import sys
sys.path.append(
    'C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/Moog_Motor_PanTilt')
sys.path.append(
    'C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/')
sys.path.append(
    'C:/Users/ULTRASIP_1/Documents/ULTRASIP/InstrumentCode/Instrument_Control/Rotation_Motor')
import Cam_Cmmand as cam
import motor_commands as mc
import elliptec

# Set Measurement Parameters
# Moog Connections
moog = serial.Serial()
moog.baudrate = 9600
moog.port = 'COM2'
moog.open()

# Pan and tilt of moog
pan = 9999
tilt = 9999

# Polarizer rotation connect and angles
controller = elliptec.Controller('COM4')
ro = elliptec.Rotator(controller)
angles = [0, 45, 90, 135]

# Image Acquisition
nb_frames = 1
exposure = 1e-6

# Position Moog
mc.init_autobaud(moog)
mc.get_status_jog(moog)
#mc.mv_to_home(moog, 0000, 0000)
time.sleep(3)

mc.mv_to_coord(moog, pan, 9999)
time.sleep(3)
mc.mv_to_coord(moog, 9999, tilt)

time.sleep(4)

# Home the rotator before usage
ro.home()
# set an offset
#offset = 0
offset = -0.025
tic = time.perf_counter()
# Loop over a list of angles and acquire for each
for angle in angles:
    ro.set_angle(angle+offset)
    angleout = ro.get_angle()
    print(angleout)

    image = cam.takeimage(nb_frames, exposure)
    
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.axes = 'off'
    plt.title(str(angleout)+'deg of polarizer')
    time.sleep(3)

toc = time.perf_counter()

print(tic-toc)
# Home and close everything
#mc.mv_to_home(moog, 0000, 0000)
moog.close()
ro.home()
ro.close()
