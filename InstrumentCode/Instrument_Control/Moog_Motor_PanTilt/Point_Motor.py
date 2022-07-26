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

time.sleep(3)
moog.open()
print(moog)

print("Moog is open?  " + str(moog.is_open))
print("Moog is writable?  " + str(moog.writable()))
time.sleep(3)

#from protocol analyzer: 02 00 31 04 00 00 00 00 35 03

moog.write(b'0in31')

time.sleep(3)

moog.close()

print('done')
