# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:02:29 2022
@author: C. M. DeLeon 

This code is to define the ULTRASIP Thorlabs rotation motor as a device and 
establish a library of commands. Adapted from https://github.com/roesel/elliptec
Note: must have Thorlabs Elliptec downloaded to the computer, 
find at https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=12829
"""

import elliptec

controller = elliptec.Controller('COM4')
ro = elliptec.Rotator(controller)

# Home the rotator before usage
ro.home()

# Loop over a list of angles and acquire for each
#for angle in [0, 45, 90, 135]:
for angle in [0]:
  ro.set_angle(angle)
  # ... acquire or perform other tasks