# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:39:13 2023

@author: C. DeLeon
This code is to simulation ULTRASIP measurements around neutral points
"""

#Import Libraries 
import numpy as np
import matplotlib.pyplot as plt

#Define W-matrix of ULTRASIP 
#The analyzer vectors are the rows of the W-matrix (pg 230 of PL&OS)
W_ultrasip = 0.5*np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])

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

#Add noise 
#Gaussian noise
mean = 0
var = 0.01
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(100,100))
gauss = gauss.reshape(100,100)
#Poisson Noise 
poisson = 0#np.random.poisson(lam=(350), size=(100, 100))

aolp = aolp + gauss + poisson
dolp = dolp + gauss + poisson

#Gray line is artifact in matplotlib
plt.imshow(dolp*100,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.ylabel('Input DoLP [%]')

plt.figure()
plt.imshow(np.degrees(aolp),cmap='hsv')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.ylabel('Input AoLP [Degrees]')

I = np.ones((100, 100));
Q = I*dolp*np.cos(2*aolp)
U = I*dolp*np.sin(2*aolp)

Stokes_in = np.array([I,Q,U])

#Find matrix of measured fluxes P = (P_0,P_90,P_45,P_135).T

p = W_ultrasip@np.reshape(Stokes_in,(3,10000))

S_out = np.linalg.pinv(W_ultrasip)@p
S_out = np.reshape(S_out,(3,100,100))

p = np.reshape(p,(4,100,100))

#Define I,Q,U
I = S_out[0]
Q = S_out[1]
U = S_out[2]

dolp_m = (np.sqrt(Q**2+U**2)/I)*100
aolp_m = np.degrees(0.5*np.arctan2(U,Q))

plt.figure()
plt.imshow(dolp_m,cmap='gray')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.ylabel('Simulated Measured DoLP [%]')

plt.figure()
plt.imshow(Q,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.ylabel('Q')

plt.figure()
plt.imshow(U,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.ylabel('U')

plt.figure()
plt.imshow(aolp_m,cmap='hsv')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.ylabel('Simulated Measured AoLP [Degrees]')


# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot data on each subplot
im1=axes[0, 0].imshow(p[0],cmap='Blues')
axes[0, 0].set_title('P_0')
#cbar = fig.colorbar(im1)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

im2=axes[0, 1].imshow(p[1],cmap='Blues')
axes[0, 1].set_title('P_90')
#cbar = fig.colorbar(im2)
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

im3=axes[1, 0].imshow(p[2],cmap='Blues')
axes[1, 0].set_title('P_45')
#cbar = fig.colorbar(im3)
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

im4 = axes[1, 1].imshow(p[3],cmap='Blues', vmin = 0, vmax = 1)
axes[1, 1].set_title('P_135')
#cbar = fig.colorbar(im4)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])


# Add a universal colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar = fig.colorbar(im4, cax=cbar_ax)


# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

