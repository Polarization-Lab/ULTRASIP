# -*- coding: utf-8 -*-
"""
This code is to calculate stokes parameters 

@author: C.M.DeLeon
"""

import cv2

0deg = cv2.imread('.png')
90deg = cv2.imread('.png')
  
# subtract the images
subtracted = cv2.subtract(0deg, 90deg)
  
# TO show the output
cv2.imshow('image', subtracted)

