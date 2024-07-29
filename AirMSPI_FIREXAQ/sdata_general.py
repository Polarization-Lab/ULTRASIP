# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:16:45 2024

@author: ULTRASIP_1
"""

#import libraries 
import os
import numpy as np
import itertools as it

outpath = "C:/Users/ULTRASIP_1/OneDrive/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/July2524"

# Change to the output directory
os.chdir(outpath) 

outfile_base = 'FWD_Scat'
        
# Generate an output file name

outfile = outfile_base+".sdat"
        

print("Saving: "+outfile)
    
# Open the output file

outputFile = open(outfile, 'w')

# Write the sdat header information

out_str = 'SDATA version 2.0\n'
outputFile.write(out_str)
out_str = '  1   1   1  : NX NY NT\n'
outputFile.write(out_str)
out_str = '\n'
outputFile.write(out_str)

sdat_date = '2024-07-29'
sdat_time = '13:19:00Z'
lon_median = -110
lat_median = 32
wave_num = 7
waves = np.array(['0.35510', '0.37720', '0.44330', '0.46910','0.55350', '0.65913', '0.86370' ]) #wavelengths in um 

out_str = '  1   '+sdat_date+'T'+sdat_time
out_str = out_str+'       60.00   0   1   : NPIXELS  TIMESTAMP  HEIGHT_OBS(m)  NSURF  IFGAS    1\n'
outputFile.write(out_str)
    
# Generate content for sdat (single line)

out_str = '           1'  # x-coordinate (ix)
out_str = out_str+'           1'  # y-coordinate (iy)
out_str = out_str+'           1'  # Cloud Flag (0=cloud, 1=clear)
out_str = out_str+'           1'  # Pixel column in grid (icol)
out_str = out_str+'           1'  # Pixel line in grid (row)


out_str = out_str+' ' + str(lon_median)  # Longitude
out_str = out_str+" " +str(lat_median) # Latitude
out_str = out_str+'{:17.8f}'.format(60) # Elevation-read from inv sdat

out_str = out_str+'      100.000000'  # Percent of land
out_str = out_str+'{:16d}'.format(wave_num)  # Number of wavelengths (nwl)
#   ## SET UP THE WAVELENGTH AND MEASUREMENT INFORMATION
for value in range(0,wave_num):
    out_str = out_str+' '+waves[value]

# Loop over the number of types of measurements per wavelength
for n in range(0,wave_num):
    out_str = out_str+'{:12d}'.format(4)


# # Loop over the measurement types per wavelength
# # NOTE: Values can be found in the GRASP documentation in Table 4.5
# #       41 = Normalized radiance (I = rad*pi/E0) - GRASP calls normalized (reduced) radiance
# #       12 = AOD, 42 = Q, 43 = U
for n in range(0,wave_num):
    out_str = out_str+'{:12d}'.format(12)
    out_str = out_str+'{:12d}'.format(41)
    out_str = out_str+'{:12d}'.format(42)
    out_str = out_str+'{:12d}'.format(43)

# # Loop over the number of measurements per wavelength per meas type
# # Note: This is the number of stares in the step-and-stare sequence
meas_num = 180

for n in range(0,wave_num):
    out_str = out_str+'{:12d}'.format(meas_num)
    out_str = out_str+'{:12d}'.format(meas_num)
    out_str = out_str+'{:12d}'.format(meas_num)
    out_str = out_str+'{:12d}'.format(meas_num)

## ANGLE DEFINITIONS

# Solar zenith angle per wavelength
# NOTE: This is per wavelength rather than per measurement (probably because of 
#       AERONET), so we take the average solar zenith angle, although this
#       varies from measurement to measurement from AirMSPI

#sun zenith/wavelength
sza=32
for w in range(0,wave_num):
    out_str = out_str+' '+str(32)

#view zenith
for y in range(12):
    for x in range(6):
        for num in range(sza,sza+30,1):
            vza = num
            out_str = out_str+' '+str(vza)

for y in range(12):
    for num in range(0,180,1):
        vaz = num
        out_str = out_str+' '+str(vaz)

for n in range(wave_num):
    out_str = out_str +'{:12f}'.format(0.22)*meas_num
    out_str = out_str +'{:12f}'.format(10)*meas_num
    out_str = out_str +'{:12f}'.format(0.1)*meas_num
    out_str = out_str +'{:12f}'.format(0.1)*meas_num


## ADDITIONAL PARAMETERS
# NOTE: This is kludgy and GRASP seems to run without this being entirely correct

out_str = out_str+'       0.00000000'  # Ground parameter (wave 1)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 2)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 3)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 4)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 5)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 6)
out_str = out_str+'       0.00000000'  # Ground parameter (wave 7)
out_str = out_str+'       0'  # Gas parameter (wave 1)
out_str = out_str+'       0'  # Gas parameter (wave 2)
out_str = out_str+'       0'  # Gas parameter (wave 3)
out_str = out_str+'       0'  # Gas parameter (wave 4)
out_str = out_str+'       0'  # Gas parameter (wave 5)
out_str = out_str+'       0'  # Gas parameter (wave 6)
out_str = out_str+'       0'  # Gas parameter (wave 7)
out_str = out_str+'       0'  # Covariance matrix (wave 1)
out_str = out_str+'       0'  # Covariance matrix (wave 2)
out_str = out_str+'       0'  # Covariance matrix (wave 3)
out_str = out_str+'       0'  # Covariance matrix (wave 4)
out_str = out_str+'       0'  # Covariance matrix (wave 5)
out_str = out_str+'       0'  # Covariance matrix (wave 6)
out_str = out_str+'       0'  # Covariance matrix (wave 7)
out_str = out_str+'       0'  # Vertical profile (wave 1)
out_str = out_str+'       0'  # Vertical profile (wave 2)
out_str = out_str+'       0'  # Vertical profile (wave 3)
out_str = out_str+'       0'  # Vertical profile (wave 4)
out_str = out_str+'       0'  # Vertical profile (wave 5)
out_str = out_str+'       0'  # Vertical profile (wave 6)
out_str = out_str+'       0'  # Vertical profile (wave 7)
out_str = out_str+'       0'  # (Dummy) (wave 1)
out_str = out_str+'       0'  # (Dummy) (wave 2)
out_str = out_str+'       0'  # (Dummy) (wave 3)
out_str = out_str+'       0'  # (Dummy) (wave 4)
out_str = out_str+'       0'  # (Dummy) (wave 5)
out_str = out_str+'       0'  # (Dummy) (wave 6)
out_str = out_str+'       0'  # (Dummy) (wave 7)
out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
out_str = out_str+'       0'  # (Extra Dummy) (wave 1)
out_str = out_str+'       0'  # (Extra Dummy) (wave 2)
out_str = out_str+'       0'  # (Extra Dummy) (wave 3)
out_str = out_str+'       0'  # (Extra Dummy) (wave 4)
out_str = out_str+'       0'  # (Extra Dummy) (wave 5)
out_str = out_str+'       0'  # (Extra Dummy) (wave 6)
out_str = out_str+'       0'  # (Extra Dummy) (wave 7)
                   
                   
# # Endline
out_str = out_str+'\n'

# Write out the line
outputFile.write(out_str)

# Close the output file
outputFile.close()        
    
    