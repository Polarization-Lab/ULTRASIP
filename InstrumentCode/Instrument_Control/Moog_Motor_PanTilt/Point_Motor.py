# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 06:48:26 2022
@author: C.M.DeLeon
Acknowledgement: Sierra Macleod 

This code is to establish precision pointing of the Moog motor using RS232 serial protocols.
Note: Must have PTCR-20 installed on computer.
"""

#Import libraries 
import serial
import time
import motor_commands as mc
from motor_commands import LimitAxis


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# tilt range: -68.4 to +68.4
def move_to_coords(pan=None, tilt=None):
    calc_pan = 9999 if pan is None else int(map_range(pan, -360, 360, 0, -622))
    calc_tilt = 9999 if tilt is None else int(map_range(tilt - 5, -68.4, 68.4, 180, -540))
    print(f'Moving to coords pan: {pan}deg (calc: {calc_pan}), tilt: {tilt}deg (calc: {calc_tilt})')
    mc.mv_to_coord(moog, calc_pan, calc_tilt)

                         
if __name__ == '__main__':
    #Configure port connection
    moog = serial.Serial()
    moog.baudrate = 9600
    moog.port = 'COM2'
    moog.open()
    #print(moog)

    #print("Moog is open?  " + str(moog.is_open))
    #print("Moog is writable?  " + str(moog.writable()))
    mc.init_autobaud(moog);

   # print('Fetching with response:')
    mc.get_status_jog(moog)
    mc.mv_to_home(moog,0000,0000)

    time.sleep(3)
 

 #coordinate must consist of the desired position to 1/10th degree 
 #multiplied by 10, i.e., +90.0° should be sent as 900.   
   
    # pan = 150;
    # tilt = 20;
    #mc.mv_to_coord(moog,pan,tilt)
    move_to_coords(tilt=-50.0)
    time.sleep(3)
    move_to_coords(tilt=0)
    time.sleep(3)
    move_to_coords(tilt=50.0)
    time.sleep(3)
    #mc.get_set_pan_tilt_soft_lims(moog, LimitAxis.CW)
    #mc.get_set_pan_tilt_soft_lims(moog, LimitAxis.CCW)
    #mc.get_set_pan_tilt_soft_lims(moog, LimitAxis.UP)
    #mc.get_set_pan_tilt_soft_lims(moog, LimitAxis.DOWN)
   
    #mc.mv_to_coord(moog,9999,-100)

    time.sleep(3)
    mc.mv_to_home(moog,0000,0000)
    

    moog.close()

    print('done')

