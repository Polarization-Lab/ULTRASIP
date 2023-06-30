# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:29:00 2023

@author: ULTRASIP_1
"""
#import libraries 
import os
import numpy as np

#header stuff, meas types/wavelength, num of valid meas/wavelength/type,sza,thetav/meas/wavelength,raz/meas/wavelength,meas/wave/type

#load in GRASP output 
# Define GRASP output file path 
outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/June2523/Washington1"



file = open(outpath+"/Merd_Inchelium.txt")
content = file.readlines()
wave_num = 7
meas_num = 9


# Change to the output directory
os.chdir(outpath) 
    
# Get the software version number to help track issues
hold = os.path.basename(__file__)
words = hold.split('_')
temp = words[len(words)-1]  # Choose the last element
hold = temp.split('.')
vers = hold[0]

outfile_base = 'RotfromMerdFWD'
        
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

for i in range(len(content)):

    if 'Date:' in content[i]:
        sdat_date = content[i].split()[1]
        
    if 'Time:' in content[i]:
        sdat_time = content[i].split()[1]
        
    if 'Longitude:' in content[i]:
        lon_median = content[i].split()[1]

    if 'Latitude' in content[i]:
        lat_median = content[i].split()[2]

out_str = '  1   '+sdat_date+'T'+sdat_time
out_str = out_str+'       70000.00   0   1   : NPIXELS  TIMESTAMP  HEIGHT_OBS(m)  NSURF  IFGAS    1\n'
outputFile.write(out_str)
    
# Generate content for sdat (single line)

out_str = '           1'  # x-coordinate (ix)
out_str = out_str+'           1'  # y-coordinate (iy)
out_str = out_str+'           1'  # Cloud Flag (0=cloud, 1=clear)
out_str = out_str+'           1'  # Pixel column in grid (icol)
out_str = out_str+'           1'  # Pixel line in grid (row)


out_str = out_str+' ' + lon_median  # Longitude
out_str = out_str+" " +lat_median # Latitude
out_str = out_str+'{:17.8f}'.format(669.31402666) # Elevation-read from inv sdat

out_str = out_str+'      100.000000'  # Percent of land
out_str = out_str+'{:16d}'.format(wave_num)  # Number of wavelengths (nwl)
#   ## SET UP THE WAVELENGTH AND MEASUREMENT INFORMATION
        
# # Loop through wavelengths

for i in range(len(content)):
    if 'Wavelength (um), AOD_' in content[i]:        
       for num in range(i+1, i+wave_num+1):
           out_str = out_str +' '+content[num].split()[0]
       
# Loop over the number of types of measurements per wavelength

out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1) # 1 meas per wave
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1)



# # Loop over the measurement types per wavelength
# # NOTE: Values can be found in the GRASP documentation in Table 4.5
# #       41 = Normalized radiance (I = rad*pi/E0) - GRASP calls normalized (reduced) radiance
# #       12 = AOD

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(12)
     
# # Loop over the number of measurements per wavelength per meas type
# # Note: This is the number of stares in the step-and-stare sequence

out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num) 
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)

## ANGLE DEFINITIONS

# Solar zenith angle per wavelength
# NOTE: This is per wavelength rather than per measurement (probably because of 
#       AERONET), so we take the average solar zenith angle, although this
#       varies from measurement to measurement from AirMSPI

for i in range(len(content)):
    if ' wavelength # ' in content[i]:    
        for n in range(3,3+wave_num):
            #print(content[i+n].split())
            out_str = out_str+' '+content[i+n].split()[1]
        for n in range(3,3+meas_num):
            #print(content[i+n].split())
            out_str = out_str+' '+content[i+n].split()[2]
        for n in range(3,3+meas_num):
            #print(content[i+n].split())
            out_str = out_str+' '+content[i+n].split()[3]

    if ', AOD_Total' in content[i]: 
        for n in range(1,1+wave_num):
           aod = ' ' + (content[i+n].split()[1])+' '
           out_str = out_str+aod*9

# for outer in range(num_meas):
#     for inner in range(num_step): 
#         out_str = out_str+'{:16.8f}'.format(vza_median[inner,outer])


# # Relative azimuth angle per measurement per wavelength
# for outer in range(num_meas):
#     for inner in range(num_step): 
#         out_str = out_str+'{:16.8f}'.format(raz_median[inner,outer])



# for i in range(len(content)):

#     if ', AOD_Total' in content[i]:
        
#         print(content[i])   
#         for num in range(i+1, i+wave_num+1):           
#             outfile = outfile + (content[num].split()[0])
    
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
                   
# # Endline
out_str = out_str+'\n'

# Write out the line
outputFile.write(out_str)

# Close the output file
outputFile.close()        
    
    