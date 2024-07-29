# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:57:32 2024

@author: ULTRASIP_1
"""
import numpy as np
import matplotlib.pyplot as plt

file = open("C:/Users/ULTRASIP_1/OneDrive/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Aug2123/rayleightest.txt","r")

lines = file.readlines()
sza = []
vza = []
vaz = []
sca = []
data = []
for x in lines:
    if '32.00' in x:
        sza = np.append(sza,float(x.split()[1]))
        vza = np.append(vza,float(x.split()[2]))
        vaz = np.append(vaz,float(x.split()[3]))
        sca = np.append(sca, float(x.split()[4]))
        data = np.append(data,float(x.split()[6]))

I = data[0:44]
Q = data[45:89]
U = data[90:134]

dolp = (np.sqrt(Q[:]**2+U[:]**2)/I[:])*100
aolp = 0.5*np.arctan2(U[:],Q[:])

sza = sza[0:44]
vza = vza[0:44]
vaz = vaz[0:44]
sca = sca[0:44]

plt.plot(sca,dolp)

file.close()