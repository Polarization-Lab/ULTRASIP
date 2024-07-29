# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:57:32 2024

@author: ULTRASIP_1
"""
import numpy as np
import matplotlib.pyplot as plt

file = open("C:/Users/ULTRASIP_1/OneDrive/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/July2524/rayleigh.txt","r")

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

I = data[0:180]
Q = data[180:360] #Q/I
U = data[360:541] #U/I

Q = Q[:]*I[:]
U = U[:]*I[:]

dolp = (np.sqrt(Q[:]**2+U[:]**2)/I[:])*100
aolp = np.degrees(0.5*np.arctan2(U[:],Q[:]))

sza = sza[0:180]
vza = vza[0:180] #view zenith angle
vaz = vaz[0:180] #view azimuth angle
sca = 180-sca[0:180]

plt.plot(sca[0:30],dolp[0:30],label='DoLP [%]')
plt.plot(sca[0:30],aolp[0:30],label='AoLP [deg]')
plt.xlabel('Scattering Angle')
plt.title('GRASP Forward Simulation Rayleigh Atmosphere-No Neutral Points')
plt.legend()

file.close()