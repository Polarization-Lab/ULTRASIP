# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:39:13 2023

@author: C. DeLeon
This code is to simulation ULTRASIP measurements around neutral points
"""

#Import Libraries 
import numpy as np
import matplotlib as plt

#Define W-matrix of ULTRASIP 
#The analyzer vectors are the rows of the W-matrix (pg 230 of PL&OS)
W_ultrasip = (1/2)*np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])

#Define input Stokes vector as a function of DoLP, AoLP, I
I = 1;
# image dimensions
xsize=100
ysize=100

# make some empty placeholder arrays to fill later
aolp = np.zeros((xsize,ysize))
dolp = aolp.copy()

for xx in range(0,xsize): # for loop over x pixel index
    for yy in range(0,ysize): # for loop over y pixel index
        #aolp[xx,yy] = np.mod(0.5*np.arctan2((yy-ysize/2)/ysize,(xx-xsize/2)/xsize),np.pi) # define AoLP with arctan, singularity at center pixel
        aolp[xx,yy] = 0.5*np.arctan2((xx-xsize/2)/xsize,(yy-ysize/2)/ysize)
        dolp[xx,yy] = np.sqrt(((xx-xsize/2)/xsize)**2+((yy-ysize/2)/ysize)**2) #define DoLP with radial distance from singularity    

#Gray line is artifact in matplotlib
plt.pyplot.imshow(dolp,cmap='twilight')
plt.pyplot.colorbar()
plt.pyplot.ylabel('Input DoLP [%]')

plt.pyplot.figure()
plt.pyplot.imshow(np.degrees(aolp),cmap='hsv')
plt.pyplot.colorbar()
plt.pyplot.ylabel('Input AoLP [Degrees]')

I = np.ones((100, 100));
Q = I*dolp*np.cos(2*aolp)
U = I*dolp*np.sin(2*aolp)

Stokes_in = np.array([I,Q,U])

#Find matrix of measured fluxes P = (P_0,P_90,P_45,P_135).T

p = W_ultrasip@np.reshape(Stokes_in,(3,10000))
p = np.reshape(p,(4,100,100))

#Define I,Q,U
I = p[0]+p[1]+p[2]+p[3]
Q = p[0]-p[1]
U = p[2]-p[3]

dolp_m = np.sqrt(Q**2+U**2)*100/I
aolp_m = np.degrees(0.5*np.arctan2(U,Q))

plt.pyplot.figure()
plt.pyplot.imshow(dolp_m,cmap='twilight')
plt.pyplot.colorbar()
plt.pyplot.ylabel('Measured DoLP [%]')

plt.pyplot.figure()
plt.pyplot.imshow(aolp_m,cmap='hsv')
plt.pyplot.colorbar()
plt.pyplot.ylabel('Measured AoLP [Degrees]')
