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
#The analyzer vectors (P0,P90,P45,P135) are the rows of the W-matrix (pg 230 of PL&OS)
W_ultrasip = 0.5*np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])

#Define input Stokes vector as a function of DoLP, AoLP, I
I = 1;
# image dimensions
xsize=256
ysize=256

# make some empty placeholder arrays to fill later
aolp = np.zeros((xsize,ysize))
dolp = aolp.copy()

for xx in range(0,xsize): # for loop over x pixel index
    for yy in range(0,ysize): # for loop over y pixel index
        aolp[xx,yy] = 0.5*np.arctan2((xx-xsize/2)/xsize,(yy-ysize/2)/ysize) 
        dolp[xx,yy] = np.sqrt(((xx-xsize/2)/xsize)**2+((yy-ysize/2)/ysize)**2) #define DoLP with radial distance from singularity    

I = np.ones((ysize, xsize));
Q = I*dolp*np.cos(2*aolp)
U = I*dolp*np.sin(2*aolp)

Stokes_in = np.array([I,Q,U])


#Plot input 
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# Plot data on each subplot
im1=axes[0, 0].imshow(I,cmap='bwr', vmin = -1, vmax = 1)
axes[0, 0].set_title('I')
cbar = fig.colorbar(im1)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

im2=axes[1, 0].imshow(Q,cmap='bwr', vmin = -1, vmax = 1)
axes[1, 0].set_title('Q')
cbar = fig.colorbar(im2)
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

im3=axes[2, 0].imshow(U,cmap='bwr',vmin = -1, vmax = 1)
axes[2, 0].set_title('U')
cbar = fig.colorbar(im3)
axes[2, 0].set_xticks([])
axes[2, 0].set_yticks([])

im4 = axes[0, 1].imshow(dolp,cmap='gray', vmin = 0, vmax = 1)
axes[0, 1].set_title('DoLP')
cbar = fig.colorbar(im4)
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

im5 = axes[1, 1].imshow(np.degrees(aolp),cmap='hsv', vmin = -90, vmax = 90)
axes[1, 1].set_title('AoLP')
cbar = fig.colorbar(im5)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

fig.delaxes(axes[2,1])

# Add figure title
fig.suptitle('Input Polarization', fontsize=20)

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()


#Find matrix of measured fluxes P = (P_0,P_90,P_45,P_135).T

p = W_ultrasip@np.reshape(Stokes_in,(3,ysize*xsize))

S_out = np.linalg.pinv(W_ultrasip)@p
S_out = np.reshape(S_out,(3,ysize,xsize))

p = np.reshape(p,(4,ysize,xsize))


grad0=np.round(np.gradient(p[0],axis=1),6)
grad3=np.round(np.gradient(p[3],axis=0),6)

grad1=np.round(np.gradient(p[1],axis=1),6)
grad2=np.round(np.gradient(p[2],axis=0),6)


#Define I,Q,U
Io = S_out[0]
Qo= S_out[1]
Uo = S_out[2]

dolp_m = (np.sqrt(Qo**2+Uo**2)/Io)
aolp_m = 0.5*np.arctan2(Uo,Qo)


#Plot output 
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# Plot data on each subplot
im1=axes[0, 0].imshow(Io,cmap='bwr', vmin = -1, vmax = 1)
axes[0, 0].set_title('I')
cbar = fig.colorbar(im1)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

im2=axes[1, 0].imshow(Qo,cmap='bwr', vmin = -1, vmax = 1)
axes[1, 0].set_title('Q')
cbar = fig.colorbar(im2)
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

im3=axes[2, 0].imshow(Uo,cmap='bwr', vmin = -1, vmax = 1)
axes[2, 0].set_title('U')
cbar = fig.colorbar(im3)
axes[2, 0].set_xticks([])
axes[2, 0].set_yticks([])

im4 = axes[0, 1].imshow(dolp_m,cmap='gray', vmin = 0, vmax = 1)
axes[0, 1].set_title('DoLP')
cbar = fig.colorbar(im4)
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

im5 = axes[1, 1].imshow(np.degrees(aolp_m),cmap='hsv', vmin = -90, vmax = 90)
axes[1, 1].set_title('AoLP')
cbar = fig.colorbar(im5)
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

fig.delaxes(axes[2,1])

# Add figure title
fig.suptitle('Output Polarization', fontsize=20)

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()

