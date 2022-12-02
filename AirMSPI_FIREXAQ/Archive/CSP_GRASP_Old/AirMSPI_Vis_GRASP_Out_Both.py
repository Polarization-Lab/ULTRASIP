# GRASP_Prescott_output_I_vis_08.py
#
# This is a Python 3.9.13 code to read the output of the GRASP retrieval code
# for AirMSPI observations over Prescott, AZ
#


# Import packages

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

#def main():  # Main code

# Set the paths
# NOTE: basepath is the location of the GRASP output files
#       figpath is where the image output should be stored

basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/"
figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Figures"

# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

num_step = 5
    
# Set the number of wavelengths for intensity

num_int = 7
    
### Create arrays to store data
# NOTE: Pay attention to the number of wavelengths

i_obs = np.zeros((num_step,num_int))  # Observed intensity
i_mod = np.zeros((num_step,num_int))  # Modeled intensity
scat = np.zeros((num_step,num_int))  # Scattering angle
vza = np.zeros((num_step,num_int))  # View zenith angle

wave = np.zeros(num_int)  # Wavelength    
aod = np.zeros(num_int)  # Aerosol optical depth
ssa = np.zeros(num_int)  # Single scattering albedo
aaod = np.zeros(num_int)  # Absorption aerosol optical depth
nr = np.zeros(num_int)  # Real part of the refractive index
ni = np.zeros(num_int)  # Imaginary part of the refractive index

# Change directory to the basepath

os.chdir(basepath)

# Get the text file listing

file_list = glob.glob('*ALL*GRASP*.txt')
    
num_files = len(file_list)
    
print("FOUND FILES:",num_files)
    
### READ DATA

# Get the correct file

inputName = file_list[0]
    
# Set a counter

data_count = 0
    
# Set some indices

sphere_next = -1

aod_start = 999
aod_count = 0
    
ssa_start = 999
ssa_count = 0
    
aaod_start = 999
aaod_count = 0
    
nr_start = 999
nr_count = 0
    
ni_start = 999
ni_count = 0
    
fit = 0

# Open the file

inputFile = open(inputName, 'r')
print("Reading: "+inputName)  # Tell user location in the process

# Get the correct file

inputName = file_list[0]
    
# Set a counter

data_count = 0
    
# Set some indices

sphere_next = -1

aod_start = 999
aod_count = 0
    
ssa_start = 999
ssa_count = 0
    
aaod_start = 999
aaod_count = 0
    
nr_start = 999
nr_count = 0
    
ni_start = 999
ni_count = 0
    
fit = 0

# Open the file

inputFile = open(inputName, 'r')
print("Reading: "+inputName)  # Tell user location in the process

# Read the data file

for line in inputFile:
    
# Check for lines with more than zero elements        
        
    if(len(line) > 1):
    
# Parse the line on blanks

        words = line.split()
            
# Check for "%" for percent of spherical particles

        if((words[0]=="%")):
            sphere_next = data_count+1
                
        if(data_count==sphere_next):
            per_sphere = float(words[1])

# Check for "WAVELENGTH" - FIRST IS FOR AOD

        if((words[0]=="wavelength")&(aod_count<1)):
                aod_start = data_count
                
# Read the AOD

        if((data_count>aod_start)&(aod_count<num_int)):
                wave[aod_count] = float(words[0])
                aod[aod_count] = float(words[1])
                aod_count = aod_count+1
            
# Check for "WAVELENGTH" - SECOND IS FOR SSA

        if((words[0]=="wavelength")&(aod_count>0)):
                ssa_start = data_count
                
# Read the SSA

        if((data_count>ssa_start)&(ssa_count<num_int)):
                ssa[ssa_count] = float(words[1])
                ssa_count = ssa_count+1
                
# Check for "WAVELENGTH" - THIRD IS FOR AAOD

        if((words[0]=="wavelength")&(ssa_count>0)):
                aaod_start = data_count
                
# Read the AAOD

        if((data_count>aaod_start)&(aaod_count<num_int)):
                aaod[aaod_count] = float(words[1])
                aaod_count = aaod_count+1

# Read the AAOD

        if((data_count>aaod_start)&(aaod_count<num_int)):
                aaod[aaod_count] = float(words[1])
                aaod_count = aaod_count+1
                
# Check for "WAVELENGTH" - FOURTH IS FOR THE REAL PART OF THE REFRACTIVE INDEX

        if((words[0]=="wavelength")&(aaod_count>0)):
                nr_start = data_count
                
# Read the real part of the refractive index

        if((data_count>nr_start)&(nr_count<num_int)):
                nr[nr_count] = float(words[1])
                nr_count = nr_count+1
                
# Check for "WAVELENGTH" - FIFTH IS FOR THE IMAGINARY PART OF THE REFRACTIVE INDEX

        if((words[0]=="wavelength")&(nr_count>0)):
                ni_start = data_count
                
# Read the imaginary part of the refractive index

        if((data_count>ni_start)&(ni_count<num_int)):
                ni[ni_count] = float(words[1])
                ni_count = ni_count+1

### GET THE FITTING RESULTS

        if(words[0]=="pixel"):
                wl = int(words[5])-1  # Wavelength index
                fit = 1  # Indicate in fitting section of results
                
        if((words[0]=="1")&(fit>0)):
                obs = int(words[0])-1
                
                vza[obs,wl] = float(words[2])
                scat[obs,wl]=float(words[4])
                i_obs[obs,wl] = float(words[5])
                i_mod[obs,wl] = float(words[6])
                    
        if((words[0]=="2")&(fit>0)):
                obs = int(words[0])-1
                
                vza[obs,wl] = float(words[2])
                scat[obs,wl]=float(words[4])
                i_obs[obs,wl] = float(words[5])
                i_mod[obs,wl] = float(words[6])
                    
        if((words[0]=="3")&(fit>0)):
                obs = int(words[0])-1
                
                vza[obs,wl] = float(words[2])
                scat[obs,wl]=float(words[4])
                i_obs[obs,wl] = float(words[5])
                i_mod[obs,wl] = float(words[6])
                    
        if((words[0]=="4")&(fit>0)):
                obs = int(words[0])-1
                
                vza[obs,wl] = float(words[2])
                scat[obs,wl]=float(words[4])
                i_obs[obs,wl] = float(words[5])
                i_mod[obs,wl] = float(words[6])
                    
        if((words[0]=="5")&(fit>0)):
                obs = int(words[0])-1
                
                vza[obs,wl] = float(words[2])
                scat[obs,wl]=float(words[4])
                i_obs[obs,wl] = float(words[5])
                i_mod[obs,wl] = float(words[6])
       
        data_count = data_count+1

# Close the input file

inputFile.close()