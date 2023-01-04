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

                         
if __name__ == '__main__':
    #Configure port connection
    moog = serial.Serial()
    moog.baudrate = 9600
    moog.port = 'COM2'

    time.sleep(3)
    moog.open()
    print(moog)

    print("Moog is open?  " + str(moog.is_open))
    print("Moog is writable?  " + str(moog.writable()))
    time.sleep(3)

    mc.init_autobaud(moog);

    print('Fetching with response:')
    mc.get_status_jog(moog)
    mc.mv_to_home(moog,0000,0000)
    time.sleep(10)

 #coordinate must consist of the desired position to 1/10th degree 
 #multiplied by 10, i.e., +90.0° should be sent as 900.   
   
    pan = 45*10;
    tilt = 10*10;
    mc.mv_to_coord(moog,pan,9999)
    time.sleep(3)
    mc.mv_to_coord(moog,9999,tilt)
    
    time.sleep(10)
    mc.mv_to_home(moog,0000,0000)

    moog.close()

    print('done')

