

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:48:23 2022

@author: ULTRASIP_1
"""

"""
Code to plot output with I and Ipol,Q,U combined and retrieved surface
parameters.
"""


# Import packages

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def main():  # Main code

# Set the paths
# NOTE: basepath is the location of the GRASP output files
#       figpath is where the image output should be stored


    # basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Mar0923/1_FIREX"
    # figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Mar0923/1_FIREX/Plots"
    
    basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Aug123/"
    #figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/July2923/"

# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5

    
# Set the number of wavelengths

    num_int = 7
    
### Create arrays to store data
# NOTE: Pay attention to the number of wavelengths

    i_obs = np.zeros((num_step,num_int))  # Observed intensity
    i_mod = np.zeros((num_step,num_int))  # Modeled intensity
    
    q_obs = np.zeros((num_step,num_int))  # Observed Q
    q_mod = np.zeros((num_step,num_int))  # Modeled Q
    
    u_obs = np.zeros((num_step,num_int))  # Observed U
    u_mod = np.zeros((num_step,num_int))  # Modeled U
    
    scat = np.zeros((num_step,num_int))  # Scattering angle
    vza = np.zeros((num_step,num_int))  # View zenith angle

    wave = np.zeros(num_int)  # Wavelength    
    aod = np.zeros(num_int)  # Aerosol optical depth
    ssa = np.zeros(num_int)  # Single scattering albedo
    aaod = np.zeros(num_int)  # Absorption aerosol optical depth
    nr = np.zeros(num_int)  # Real part of the refractive index
    ni = np.zeros(num_int)  # Imaginary part of the refractive index

    aae = np.zeros(1)
    
    sd = np.array([])
    radius = np.array([])
# Change directory to the basepath

    os.chdir(basepath)

# Get the text file listing

    file_list = glob.glob('Merd_R1_FWD.txt')
    
    num_files = len(file_list)
    
    print("FOUND FILES:",num_files)
    
### READ DATA
# NOTE: Changed the approach just to count the number of lines to store the
#       appropriate data

# Get the correct file

    inputName = file_list[0]
    
# Set a counter

    data_count = 0
    
# Set some other counters

    nobs = 0

# Open the file

    inputFile = open(inputName, 'r')
    print("Reading: "+inputName)  # Tell user location in the process

# Read the data file
    i=0
    for line in inputFile:
        
        #if data_count in range(48,93): #SD
        #if data_count in range(116,123):    #SSA
        # if data_count in range(108,115):    #AOD
        # if data_count in range(132,139):    #nr
        #if data_count in range(140,147):    #ni
        # if data_count in range(140,147):    #nr
        #     words = line.split()
        #     print(words[0])
        #     radius = np.append(radius,float(words[0]))
        #     sd = np.append(sd,float(words[1]))
        if data_count in range(242,247):    #dolp
            words = line.split()
            radius = np.append(radius,float(words[6]))
        if data_count in range(248,253):    #dolp
            words = line.split()
            sd = np.append(sd,float(words[6]))

        data_count = data_count+1

# # Close the input file
#     print("DONE")
#     print(sd)
#     inputFile.close()                

#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)

#     ax1.plot(radius[:],sd[:],color="blue")

#     #print(i_mod[:,3])
#     #print(i_obs[:,3])
#     # print(scat[:,6])
#     # ax1.set_xlim(60,180)
#     # ax1.set_xticks(np.arange(60,190,30))
#     ax1.set_xlabel("Radius [um]")
#     ax1.set_xscale('log')

#     # ax1.set_ylim(-0.1,0.4)
#     # ax1.set_yticks(np.arange(-0.1,0.5,0.1))
#     ax1.set_ylabel('Normalized Size Distribution')
    # ax1.legend(loc=1)  # Upper right


# Generate the output file name
    
    # hold = inputName.split(".")
    # out_base = hold[0]
    # outfile = out_base+'_VIS_v'+'_sd.png'
    # print("Saving: "+outfile)
    # plt.savefig(outfile,dpi=300) 

# # Show the plot
    
#     plt.show() 
    
#     plt.close()
    
    return radius[:],sd[:]

#Tell user completion was successful
print("\nSuccessful Completion\n")

### END MAIN FUNCTION


if __name__ == '__main__':
   fit_Q,fit_U =  main() 
   
def calculate_DoLP(Q, U):
    DoLP = np.sqrt(np.array(Q)**2 + np.array(U)**2) 
    return DoLP

def calculate_AoLP(Q, U):
    AoLP = (1/2)*np.degrees(np.arctan(np.array(U)/np.array(Q)))
    return AoLP

# Calculate DoLP for each pair of Q and U
DoLP_values = []
# AoLP_values = []
for n in range(5):
    DoLP_values.append(np.round(calculate_DoLP(fit_Q[n], fit_U[n]),3))
    # AoLP_values.append(calculate_AoLP(fit_Q[n], fit_U[n]))   
   