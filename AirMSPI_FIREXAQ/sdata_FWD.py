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

outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Aug2023"

#outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Aug2023/"


file = open(outpath+"/Merd_R5_INV.txt")
content = file.readlines()
wave_num = 7
meas_num = 45


# Change to the output directory
os.chdir(outpath) 
    
# Get the software version number to help track issues
hold = os.path.basename(__file__)
words = hold.split('_')
temp = words[len(words)-1]  # Choose the last element
hold = temp.split('.')
vers = hold[0]

outfile_base = 'R5v8-ScatFWD'
        
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
out_str = out_str+'       60.00   0   1   : NPIXELS  TIMESTAMP  HEIGHT_OBS(m)  NSURF  IFGAS    1\n'
outputFile.write(out_str)
    
# Generate content for sdat (single line)

out_str = '           1'  # x-coordinate (ix)
out_str = out_str+'           1'  # y-coordinate (iy)
out_str = out_str+'           1'  # Cloud Flag (0=cloud, 1=clear)
out_str = out_str+'           1'  # Pixel column in grid (icol)
out_str = out_str+'           1'  # Pixel line in grid (row)


out_str = out_str+' ' + lon_median  # Longitude
out_str = out_str+" " +lat_median # Latitude
out_str = out_str+'{:17.8f}'.format(60) # Elevation-read from inv sdat

out_str = out_str+'      100.000000'  # Percent of land
out_str = out_str+'{:16d}'.format(wave_num)  # Number of wavelengths (nwl)
#   ## SET UP THE WAVELENGTH AND MEASUREMENT INFORMATION
        
# # Loop through wavelengths

for i in range(len(content)):
    if 'Wavelength (um), AOD_' in content[i]:        
       for num in range(i+1, i+wave_num+1):
           out_str = out_str +' '+content[num].split()[0]
       
# Loop over the number of types of measurements per wavelength

out_str = out_str+'{:12d}'.format(4)
out_str = out_str+'{:12d}'.format(4)
out_str = out_str+'{:12d}'.format(4)
out_str = out_str+'{:12d}'.format(4) # 1 meas per wave
out_str = out_str+'{:12d}'.format(4)
out_str = out_str+'{:12d}'.format(4)
out_str = out_str+'{:12d}'.format(4)



# # Loop over the measurement types per wavelength
# # NOTE: Values can be found in the GRASP documentation in Table 4.5
# #       41 = Normalized radiance (I = rad*pi/E0) - GRASP calls normalized (reduced) radiance
# #       12 = AOD, 42 = Q, 43 = U

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)

out_str = out_str+'{:12d}'.format(12)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)


     
# # Loop over the number of measurements per wavelength per meas type
# # Note: This is the number of stares in the step-and-stare sequence

out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)


out_str = out_str+'{:12d}'.format(meas_num) 
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)


out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)


out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)


out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)
out_str = out_str+'{:12d}'.format(meas_num)


out_str = out_str+'{:12d}'.format(meas_num)
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

#sun zenith/wavelength
for i in range(len(content)):
    if 'sza' in content[i]: 
        #sza = ' ' + content[i+1].split()[1]
        sza = ' ' + str(25)
out_str = out_str+' '+sza*7


#view angle/meas
# for i in range(len(content)):
#     if 'vis' in content[i]:
#         vza1 = ' ' + content[i+1].split()[2]
#         vza2 = ' ' + content[i+2].split()[2]
#         vza3 = ' ' + content[i+3].split()[2]
#         vza4 = ' ' + content[i+4].split()[2]
#         vza5 = ' ' + content[i+5].split()[2]
#         # vza6 = ' ' + content[i+6].split()[2]
#         # vza7 = ' ' + content[i+7].split()[2]
#         # vza8 = ' ' + content[i+8].split()[2]
#         # vza9 = ' ' + content[i+9].split()[2]
        
#         vza = vza1+vza2+vza3+vza4+vza5 #+vza6+vza7+vza8+vza9
#         vza = vza*28

# Using a list comprehension to generate the numbers in the specified range
numbers = [str(num) for num in range(0,90,2)]
# Joining the numbers with a comma separator to create the final string
vza = " ".join(numbers) + ' '
# Repeating the sequence 28 times
vza = vza * 28

out_str = out_str+' '+vza

#relative azimuth/meas
# for i in range(len(content)):
#     if 'fis' in content[i]:
#         raz1 = ' ' + content[i+1].split()[3]
#         raz2 = ' ' + content[i+2].split()[3]
#         raz3 = ' ' + content[i+3].split()[3]
#         raz4 = ' ' + content[i+4].split()[3]
#         raz5 = ' ' + content[i+5].split()[3]
#         # raz6 = ' ' + content[i+6].split()[3]
#         # raz7 = ' ' + content[i+7].split()[3]
#         # raz8 = ' ' + content[i+8].split()[3]
#         # raz9 = ' ' + content[i+9].split()[3]
        
#         raz = raz1+raz2+raz3+raz4+raz5 #+raz6+raz7+raz8+raz9
#         raz = raz*28

# Using a list comprehension to generate the numbers in the specified range
numbers = [str(180-num) for num in range(0, 180, 4)]
# Joining the numbers with a comma separator to create the final string
raz = " ".join(numbers) + ' '
# Repeating the sequence 28 times
raz = raz * 28

out_str = out_str+' '+raz
        
for i in range(len(content)):
    if ', AOD' in content[i]:
        out_str = out_str+'{:12f}'.format(float(content[i+1].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+2].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+3].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+4].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+5].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+6].split()[1]))*meas_num
        out_str = out_str +'{:12f}'.format(10)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num
        out_str = out_str +'{:12f}'.format(0.1)*meas_num


        out_str = out_str+'{:12f}'.format(float(content[i+7].split()[1]))*meas_num
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
    
    