# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 06:48:26 2022
@author: C.M.DeLeon

This code is to establish precision pointing of the Moog motor using RS232 serial protocols.
Note: Must have PTCR-20 installed on computer.
"""

#Import libraries 
import serial
import time

#Configure port connection
moog = serial.Serial()
moog.baudrate = 9600
moog.port = 'COM2'
print(moog)
time.sleep(3)
moog.open()
print("Moog is open?  " + str(moog.is_open))
time.sleep(3)

moog.write(b'\31')


moog.close()

print('done')
