# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:27:12 2023

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


    #basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/Bakersfield"
    #figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/Bakersfield/Plots"

    basepath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/1FIREXR2"
    figpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/1FIREXR2/Plots"

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
    
    sd = np.zeros(45)
    radius = np.zeros(45)
# Change directory to the basepath

    os.chdir(basepath)

# Get the text file listing

    file_list = glob.glob('Scat*.txt')
    
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

    for line in inputFile:
             
        if(data_count==96):  # Sphericity
            words = line.split()
            print(words)
            per_sphere = float(words[1])
            
        if(data_count==100):  # Sphericity
            words = line.split()
            print(words)
            aae = float(words[0])
            
        if(data_count==102):  # AOD
            words = line.split()
            print(words)
            wave[0] = float(words[0])
            aod[0] = float(words[1])
            
        if(data_count==103):
            words = line.split()
            print(words)
            wave[1] = float(words[0])
            aod[1] = float(words[1])
        
        if(data_count==104):
            words = line.split()
            print(words)
            wave[2] = float(words[0])
            aod[2] = float(words[1])

        if(data_count==105):
            words = line.split()
            print(words)
            wave[3] = float(words[0])
            aod[3] = float(words[1])
            
        if(data_count==106):
            words = line.split()
            print(words)
            wave[4] = float(words[0])
            aod[4] = float(words[1])
        
        if(data_count==107):
            words = line.split()
            print(words)
            wave[5] = float(words[0])
            aod[5] = float(words[1])
            
        if(data_count==108):
            words = line.split()
            print(words)
            wave[6] = float(words[0])
            aod[6] = float(words[1])
            
        if(data_count==110):
            words = line.split()
            print(words)
            ssa[0] = float(words[1])
            
        if(data_count==111):
            words = line.split()
            print(words)
            ssa[1] = float(words[1])
        
        if(data_count==112):
            words = line.split()
            print(words)
            ssa[2] = float(words[1])

        if(data_count==113):
            words = line.split()
            print(words)
            ssa[3] = float(words[1])
        
        if(data_count==114):
            words = line.split()
            print(words)
            ssa[4] = float(words[1])  
            
        if(data_count==115):
            words = line.split()
            print(words)
            ssa[5] = float(words[1])
        
        if(data_count==116):
            words = line.split()
            print(words)
            ssa[6] = float(words[1])
            
        if(data_count==126):  # N_r
            words = line.split()
            print(words)
            nr[0] = float(words[1])

        if(data_count==127):  # N_r
            words = line.split()
            print(words)
            nr[1] = float(words[2])
            
        if(data_count==128):  # N_r
            words = line.split()
            print(words)
            nr[2] = float(words[2])
            
        if(data_count==129):  # N_r
            words = line.split()
            print(words)
            nr[3] = float(words[2])
            
        if(data_count==130):  # N_r
            words = line.split()
            print(words)
            nr[4] = float(words[2])
            
        if(data_count==131):  # N_r
            words = line.split()
            print(words)
            nr[5] = float(words[2])
        
        if(data_count==132):  # N_r
            words = line.split()
            print(words)
            nr[6] = float(words[2])
        
        if(data_count==134):  # N_i
            words = line.split()
            print(words)
            ni[0] = float(words[1])
            
        if(data_count==135):  # N_i
            words = line.split()
            print(words)
            ni[1] = float(words[1])

        if(data_count==136):  # N_i
            words = line.split()
            print(words)
            ni[2] = float(words[1])

        if(data_count==137):  # N_i
            words = line.split()
            print(words)
            ni[3] = float(words[1])
            
        if(data_count==138):  # N_i
            words = line.split()
            print(words)
            ni[4] = float(words[1])
            
        if(data_count==139):  # N_i
            words = line.split()
            print(words)
            ni[5] = float(words[1])
            
        if(data_count==140):  # N_i
            words = line.split()
            print(words)
            ni[6] = float(words[1])
            
        if(data_count==197):  #Fitting results
            words = line.split()
            print(words)
            vza[0,0] = float(words[2])
            scat[0,0]=float(words[4])
            i_obs[0,0] = float(words[5])
            i_mod[0,0] = float(words[6])
            
        if(data_count==198):  # Fitting results
            words = line.split()
            print(words)
            vza[1,0] = float(words[2])
            scat[1,0]=float(words[4])
            i_obs[1,0] = float(words[5])
            i_mod[1,0] = float(words[6])
            
        if(data_count==199):  # Fitting results I, wvl 1
            words = line.split()
            print(words)
            vza[2,0] = float(words[2])
            scat[2,0]=float(words[4])
            i_obs[2,0] = float(words[5])
            i_mod[2,0] = float(words[6])
            
        if(data_count==200):  # Fitting results, I, wvl 1
            words = line.split()
            print(words)
            vza[3,0] = float(words[2])
            scat[3,0]=float(words[4])
            i_obs[3,0] = float(words[5])
            i_mod[3,0] = float(words[6])
            
        if(data_count==201):  # Fitting results
            words = line.split()
            print(words)
            vza[4,0] = float(words[2])
            scat[4,0]=float(words[4])
            i_obs[4,0] = float(words[5])
            i_mod[4,0] = float(words[6])

        if(data_count==206):  #Fitting results, wvl 2
            words = line.split()
            print(words)
            vza[0,1] = float(words[2])
            scat[0,1]=float(words[4])
            i_obs[0,1] = float(words[5])
            i_mod[0,1] = float(words[6])
            
        if(data_count==207):  # Fitting results
            words = line.split()
            print(words)
            vza[1,1] = float(words[2])
            scat[1,1]=float(words[4])
            i_obs[1,1] = float(words[5])
            i_mod[1,1] = float(words[6])
            
        if(data_count==208):  # Fitting results I, wvl 2
            words = line.split()
            print(words)
            vza[2,1] = float(words[2])
            scat[2,1]=float(words[4])
            i_obs[2,1] = float(words[5])
            i_mod[2,1] = float(words[6])
            
        if(data_count==209):  # Fitting results, I, wvl 2
            words = line.split()
            print(words)
            vza[3,1] = float(words[2])
            scat[3,1]=float(words[4])
            i_obs[3,1] = float(words[5])
            i_mod[3,1] = float(words[6])
            
        if(data_count==210):  # Fitting results
            words = line.split()
            print(words)
            vza[4,1] = float(words[2])
            scat[4,1]=float(words[4])
            i_obs[4,1] = float(words[5])
            i_mod[4,1] = float(words[6])
            
        if(data_count==215):  #Fitting results, wvl 3
            words = line.split()
            print(words)
            vza[0,2] = float(words[2])
            scat[0,2]=float(words[4])
            i_obs[0,2] = float(words[5])
            i_mod[0,2] = float(words[6])
            
        if(data_count==216):  # Fitting results
            words = line.split()
            print(words)
            vza[1,2] = float(words[2])
            scat[1,2]=float(words[4])
            i_obs[1,2] = float(words[5])
            i_mod[1,2] = float(words[6])
            
        if(data_count==217):  # Fitting results
            words = line.split()
            print(words)
            vza[2,2] = float(words[2])
            scat[2,2]=float(words[4])
            i_obs[2,2] = float(words[5])
            i_mod[2,2] = float(words[6])
            
        if(data_count==218):  # Fitting results
            words = line.split()
            print(words)
            vza[3,2] = float(words[2])
            scat[3,2]=float(words[4])
            i_obs[3,2] = float(words[5])
            i_mod[3,2] = float(words[6])
            
        if(data_count==219):  # Fitting results
            words = line.split()
            print(words)
            vza[4,2] = float(words[2])
            scat[4,2]=float(words[4])
            i_obs[4,2] = float(words[5])
            i_mod[4,2] = float(words[6])
            
        if(data_count==224):  #Fitting results, wvl 4 (POL)
            words = line.split()
            print(words)
            vza[0,3] = float(words[2])
            scat[0,3]=float(words[4])
            i_obs[0,3] = float(words[5])
            i_mod[0,3] = float(words[6])
            
        if(data_count==225):  # Fitting results
            words = line.split()
            print(words)
            vza[1,3] = float(words[2])
            scat[1,3]=float(words[4])
            i_obs[1,3] = float(words[5])
            i_mod[1,3] = float(words[6])
            
        if(data_count==226):  # Fitting results
            words = line.split()
            print(words)
            vza[2,3] = float(words[2])
            scat[2,3]=float(words[4])
            i_obs[2,3] = float(words[5])
            i_mod[2,3] = float(words[6])
            
        if(data_count==227):  # Fitting results
            words = line.split()
            print(words)
            vza[3,3] = float(words[2])
            scat[3,3]=float(words[4])
            i_obs[3,3] = float(words[5])
            i_mod[3,3] = float(words[6])
            
        if(data_count==228):  # Fitting results
            words = line.split()
            print(words)
            vza[4,3] = float(words[2])
            scat[4,3]=float(words[4])
            i_obs[4,3] = float(words[5])
            i_mod[4,3] = float(words[6])

        if(data_count==230):  # Q
            words = line.split()
            print(words)
            q_obs[0,0] = float(words[5])
            q_mod[0,0] = float(words[6])
            
        if(data_count==231):  # Q
            words = line.split()
            print(words)
            q_obs[1,0] = float(words[5])
            q_mod[1,0] = float(words[6])
        
        if(data_count==232):  # Q
            words = line.split()
            print(words)
            q_obs[2,0] = float(words[5])
            q_mod[2,0] = float(words[6])

        if(data_count==233):  # Q
            words = line.split()
            print(words)
            q_obs[3,0] = float(words[5])
            q_mod[3,0] = float(words[6])

        if(data_count==234):  # Q
            words = line.split()
            print(words)
            q_obs[4,0] = float(words[5])
            q_mod[4,0] = float(words[6])
            
            
        if(data_count==236):  # U
            words = line.split()
            print(words)
            u_obs[0,0] = float(words[5])
            u_mod[0,0] = float(words[6])
            
        if(data_count==237):  # U
            words = line.split()
            print(words)
            u_obs[1,0] = float(words[5])
            u_mod[1,0] = float(words[6])
        
        if(data_count==238):  # U
            words = line.split()
            print(words)
            u_obs[2,0] = float(words[5])
            u_mod[2,0] = float(words[6])

        if(data_count==239):  # U
            words = line.split()
            print(words)
            u_obs[3,0] = float(words[5])
            u_mod[3,0] = float(words[6])

        if(data_count==240):  # U
            words = line.split()
            print(words)
            u_obs[4,0] = float(words[5])
            u_mod[4,0] = float(words[6])
            
        if(data_count==245):  #Fitting results, wvl 5
            words = line.split()
            print(words)
            vza[0,4] = float(words[2])
            scat[0,4]=float(words[4])
            i_obs[0,4] = float(words[5])
            i_mod[0,4] = float(words[6])
            
        if(data_count==246):  # Fitting results
            words = line.split()
            print(words)
            vza[1,4] = float(words[2])
            scat[1,4]=float(words[4])
            i_obs[1,4] = float(words[5])
            i_mod[1,4] = float(words[6])
            
        if(data_count==247):  # Fitting results
            words = line.split()
            print(words)
            vza[2,4] = float(words[2])
            scat[2,4]=float(words[4])
            i_obs[2,4] = float(words[5])
            i_mod[2,4] = float(words[6])
            
        if(data_count==248):  # Fitting results,
            words = line.split()
            print(words)
            vza[3,4] = float(words[2])
            scat[3,4]=float(words[4])
            i_obs[3,4] = float(words[5])
            i_mod[3,4] = float(words[6])
            
        if(data_count==249):  # Fitting results
            words = line.split()
            print(words)
            vza[4,4] = float(words[2])
            scat[4,4]=float(words[4])
            i_obs[4,4] = float(words[5])
            i_mod[4,4] = float(words[6])
            
        if(data_count==254):  #Fitting results, wvl 6 (POL)
            words = line.split()
            print(words)
            vza[0,5] = float(words[2])
            scat[0,5]=float(words[4])
            i_obs[0,5] = float(words[5])
            i_mod[0,5] = float(words[6])
            
        if(data_count==255):  # Fitting results
            words = line.split()
            print(words)
            vza[1,5] = float(words[2])
            scat[1,5]=float(words[4])
            i_obs[1,5] = float(words[5])
            i_mod[1,5] = float(words[6])
            
        if(data_count==256):  # Fitting results
            words = line.split()
            print(words)
            vza[2,5] = float(words[2])
            scat[2,5]=float(words[4])
            i_obs[2,5] = float(words[5])
            i_mod[2,5] = float(words[6])
            
        if(data_count==257):  # Fitting results
            words = line.split()
            print(words)
            vza[3,5] = float(words[2])
            scat[3,5]=float(words[4])
            i_obs[3,5] = float(words[5])
            i_mod[3,5] = float(words[6])
            
        if(data_count==258):  # Fitting results
            words = line.split()
            print(words)
            vza[4,5] = float(words[2])
            scat[4,5]=float(words[4])
            i_obs[4,5] = float(words[5])
            i_mod[4,5] = float(words[6])

        if(data_count==260):  # Q
            words = line.split()
            print(words)
            q_obs[0,1] = float(words[5])
            q_mod[0,1] = float(words[6])
            
        if(data_count==261):  # Q
            words = line.split()
            print(words)
            q_obs[1,1] = float(words[5])
            q_mod[1,1] = float(words[6])
        
        if(data_count==262):  # Q
            words = line.split()
            print(words)
            q_obs[2,1] = float(words[5])
            q_mod[2,1] = float(words[6])

        if(data_count==263):  # Q
            words = line.split()
            print(words)
            q_obs[3,1] = float(words[5])
            q_mod[3,1] = float(words[6])

        if(data_count==264):  # Q
            words = line.split()
            print(words)
            q_obs[4,1] = float(words[5])
            q_mod[4,1] = float(words[6])
            
            
        if(data_count==266):  # U
            words = line.split()
            print(words)
            u_obs[0,1] = float(words[5])
            u_mod[0,1] = float(words[6])
            
        if(data_count==267):  # U
            words = line.split()
            print(words)
            u_obs[1,1] = float(words[5])
            u_mod[1,1] = float(words[6])
        
        if(data_count==268):  # U
            words = line.split()
            print(words)
            u_obs[2,1] = float(words[5])
            u_mod[2,1] = float(words[6])

        if(data_count==269):  # U
            words = line.split()
            print(words)
            u_obs[3,1] = float(words[5])
            u_mod[3,1] = float(words[6])

        if(data_count==270):  # U
            words = line.split()
            print(words)
            u_obs[4,1] = float(words[5])
            u_mod[4,1] = float(words[6])
            
        if(data_count==275):  #Fitting results, wvl 7 (POL)
            words = line.split()
            print(words)
            vza[0,6] = float(words[2])
            scat[0,6]=float(words[4])
            i_obs[0,6] = float(words[5])
            i_mod[0,6] = float(words[6])
            
        if(data_count==276):  # Fitting results
            words = line.split()
            print(words)
            vza[1,6] = float(words[2])
            scat[1,6]=float(words[4])
            i_obs[1,6] = float(words[5])
            i_mod[1,6] = float(words[6])
            
        if(data_count==277):  # Fitting results
            words = line.split()
            print(words)
            vza[2,6] = float(words[2])
            scat[2,6]=float(words[4])
            i_obs[2,6] = float(words[5])
            i_mod[2,6] = float(words[6])
            
        if(data_count==278):  # Fitting results
            words = line.split()
            print(words)
            vza[3,6] = float(words[2])
            scat[3,6]=float(words[4])
            i_obs[3,6] = float(words[5])
            i_mod[3,6] = float(words[6])
            
        if(data_count==279):  # Fitting results
            words = line.split()
            print(words)
            vza[4,6] = float(words[2])
            scat[4,6]=float(words[4])
            i_obs[4,6] = float(words[5])
            i_mod[4,6] = float(words[6])

        if(data_count==281):  # Q
            words = line.split()
            print(words)
            q_obs[0,2] = float(words[5])
            q_mod[0,2] = float(words[6])
            
        if(data_count==282):  # Q
            words = line.split()
            print(words)
            q_obs[1,2] = float(words[5])
            q_mod[1,2] = float(words[6])
        
        if(data_count==283):  # Q
            words = line.split()
            print(words)
            q_obs[2,2] = float(words[5])
            q_mod[2,2] = float(words[6])

        if(data_count==284):  # Q
            words = line.split()
            print(words)
            q_obs[3,2] = float(words[5])
            q_mod[3,2] = float(words[6])

        if(data_count==285):  # Q
            words = line.split()
            print(words)
            q_obs[4,2] = float(words[5])
            q_mod[4,2] = float(words[6])
            
            
        if(data_count==287):  # U
            words = line.split()
            print(words)
            u_obs[0,2] = float(words[5])
            u_mod[0,2] = float(words[6])
            
        if(data_count==288):  # U
            words = line.split()
            #print(words)
            u_obs[1,2] = float(words[5])
            u_mod[1,2] = float(words[6])
        
        if(data_count==289):  # U
            words = line.split()
            #print(words)
            u_obs[2,2] = float(words[5])
            u_mod[2,2] = float(words[6])

        if(data_count==290):  # U
            words = line.split()
            #print(words)
            u_obs[3,2] = float(words[5])
            u_mod[3,2] = float(words[6])

        if(data_count==291):  # U
            words = line.split()
            #print(words)
            u_obs[4,2] = float(words[5])
            u_mod[4,2] = float(words[6])

        data_count = data_count+1

# Close the input file
    print("DONE")
    inputFile.close()

### PLOT THE DATA

# Change to the figure directory

    os.chdir(figpath)
    
    fig, (ax1,ax2,ax3) = plt.subplots(
         nrows=3, ncols=1, dpi=240,figsize=(6, 8))
    
    # ax1.scatter(scat[:,0],i_obs[:,0],marker='x',color="green",label="355nm")
    # ax1.scatter(scat[:,1],i_obs[:,1],marker='x',color="pink",label="380nm")
    # ax1.scatter(scat[:,2],i_obs[:,2],marker='x',color="brown",label="445nm")  
    # ax1.scatter(scat[:,4],i_obs[:,4],marker='x',color="purple",label="555nm")
    
    ax1.plot(scat[:,0],i_obs[:,0],linestyle='dashed',color="green",label="355nm")
    ax1.plot(scat[:,1],i_obs[:,1],linestyle='dashed',color="pink",label="380nm")
    ax1.plot(scat[:,2],i_obs[:,2],linestyle='dashed',color="brown",label="445nm")  
    ax1.plot(scat[:,4],i_obs[:,4],linestyle='dashed',color="purple",label="555nm")
    

    ax1.plot(scat[:,0],i_mod[:,0],color="green")
    ax1.plot(scat[:,1],i_mod[:,1],color="pink")
    ax1.plot(scat[:,2],i_mod[:,2],color="brown")
    ax1.plot(scat[:,4],i_mod[:,4],color="purple")

    # ax1.scatter(scat[:,3],i_obs[:,3],marker='x',color="blue",label="470nm")
    # ax1.scatter(scat[:,5],i_obs[:,5],marker='x',color="red",label="660nm")
    # ax1.scatter(scat[:,6],i_obs[:,6],marker='x',color="orange",label="865nm")
    
    ax1.plot(scat[:,3],i_obs[:,3],linestyle='dashed',color="blue",label="470nm")
    ax1.plot(scat[:,5],i_obs[:,5],linestyle='dashed',color="red",label="660nm")
    ax1.plot(scat[:,6],i_obs[:,6],linestyle='dashed',color="orange",label="865nm")

    ax1.plot(scat[:,3],i_mod[:,3],color="blue")
    ax1.plot(scat[:,5],i_mod[:,5],color="red")
    ax1.plot(scat[:,6],i_mod[:,6],color="orange")
    
    # ymin = 0;
    # ymax = 0.4;
    # yticks = 0.1;
     
    #ax1.set_ylim(ymin,ymax)
    #ax1.set_yticks(np.arange(ymin,ymax,yticks))
    ax1.get_yaxis().get_major_formatter().labelOnlyBase = True
    ax1.set_yscale("log")
    # ax1.set_ylim(5e-2,2.5e-1)
    # ax1.set_yticks(np.arange(0.03,0.3,0.06))
    # ax1.set_ylabel('BRF(I)', fontsize='xx-large')
    ax1.yaxis.set_label_coords(-.15, .5)
    ax1.xaxis.set_label_coords(.5, -.1)
    ax1.set_xticks([])
    
    # ax1.legend(loc='best',ncol=4,fontsize='medium')  # Upper right
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.5,3),
    #       ncol=3, fancybox=True, shadow=True,fontsize='large')
    
    ax2.plot(scat[:,3],q_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    ax2.plot(scat[:,5],q_obs[:,1],linestyle='dashed',color="red",label="660nm")
    ax2.plot(scat[:,6],q_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    
    ax2.plot(scat[:,3],q_mod[:,0],color="blue")
    ax2.plot(scat[:,5],q_mod[:,1],color="red")
    ax2.plot(scat[:,6],q_mod[:,2],color="orange")
         
    ymin = 0.0;
    ymax = 0.5;
    yticks = 0.1;
     
    ax2.set_ylim(ymin,ymax)
    ax2.set_yticks(np.arange(ymin,ymax,yticks))
    # ax2.set_ylabel('BRF(Q)', fontsize='xx-large')
    ax2.yaxis.set_label_coords(-.15, .5)
    ax2.xaxis.set_label_coords(.5, -.1)
    ax2.set_xticks([])
    
    # ax3.scatter(scat[:,3],u_obs[:,0],marker='x',color="blue",label="470nm")
    # ax3.scatter(scat[:,5],u_obs[:,1],marker='x',color="red",label="660nm")
    # ax3.scatter(scat[:,6],u_obs[:,2],marker='x',color="orange",label="865nm")
    
    ax3.plot(scat[:,3],u_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    ax3.plot(scat[:,5],u_obs[:,1],linestyle='dashed',color="red",label="660nm")
    ax3.plot(scat[:,6],u_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    
    ax3.plot(scat[:,3],u_mod[:,0],color="blue")
    ax3.plot(scat[:,5],u_mod[:,1],color="red")
    ax3.plot(scat[:,6],u_mod[:,2],color="orange")
         
    ymin = -0.06;
    ymax =0.01;
    yticks = 0.02;
     
    ax3.set_ylim(ymin,ymax)
    ax3.set_yticks(np.arange(ymin,ymax,yticks))
    # ax3.set_ylabel('BRF(U)', fontsize='xx-large')
    
    # ax3.set_xlim(60,145)
    # ax3.set_xticks(np.arange(60,145,10))
    ax3.set_xlim(65,150)
    ax3.set_xticks(np.arange(65,150,10))
    ax3.set_xlabel( u"\u03A9 [\u00b0]",fontsize='xx-large')
    
    ax3.yaxis.set_label_coords(-.15, .5)
    ax3.xaxis.set_label_coords(.5, -.1)
   
    plt.tight_layout()  
    plt.show() 
    
    fig, ax = plt.subplots(3, 3,
                           sharex=True, 
                           sharey='row')

    ax[0,0].plot(scat[:,0],i_obs[:,0],linestyle='dashed',color="green")
    ax[0,0].plot(scat[:,1],i_obs[:,1],linestyle='dashed',color="pink")
    ax[0,0].plot(scat[:,2],i_obs[:,2],linestyle='dashed',color="brown")  
    ax[0,0].plot(scat[:,4],i_obs[:,4],linestyle='dashed',color="purple")
    
    ax[0,0].set_ylabel('BRF(I)', fontsize='x-large')
    ax[0,0].yaxis.set_label_coords(-.4, .5)
    
    ax[0,0].plot(scat[:,0],i_mod[:,0],color="green",label="355")
    ax[0,0].plot(scat[:,1],i_mod[:,1],color="pink",label="380")
    ax[0,0].plot(scat[:,2],i_mod[:,2],color="brown",label="445")
    ax[0,0].plot(scat[:,4],i_mod[:,4],color="purple",label="555")
    
    ax[0,0].set_yticks([0.06,0.1,0.2])
    
    ax[0,1].plot(scat[:,3],i_obs[:,3],linestyle='dashed',color="blue")

    ax[0,1].plot(scat[:,5],i_obs[:,5],linestyle='dashed',color="red")
    ax[0,1].plot(scat[:,6],i_obs[:,6],linestyle='dashed',color="orange")

    ax[0,1].plot(scat[:,3],i_mod[:,3],color="blue",label='470')
    ax[0,1].plot(scat[:,5],i_mod[:,5],color="red",label='660')
    ax[0,1].plot(scat[:,6],i_mod[:,6],color="orange",label='865')
    
    ax[0,0].legend(bbox_to_anchor=(3.62,1),
          ncol=2,fontsize='medium',frameon=False)
    ax[0,1].legend(bbox_to_anchor=(1.08,0.5),
          ncol=2,fontsize='medium',frameon=False)
    
    fig.delaxes(ax[0,2])
    
    ax[2,0].set_ylabel('BRF(U)', fontsize='x-large')
    ax[2,0].yaxis.set_label_coords(-.4, .5)
    
    ax[2,0].plot(scat[:,3],u_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    ax[2,0].fill_between(np.sort(scat[:,3]), u_obs[:,0]-np.std(u_obs[:,0]), u_obs[:,0]+np.std(u_obs[:,0]),color="blue",alpha=0.2)

    ax[2,1].plot(scat[:,5],u_obs[:,1],linestyle='dashed',color="red",label="660nm")
    ax[2,1].fill_between(scat[:,5], u_obs[:,1]-np.std(u_obs[:,1]), u_obs[:,1]+np.std(u_obs[:,1]),color="red",alpha=0.2)
    print(np.std(u_obs[:,1]))
    ax[2,2].plot(scat[:,6],u_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    ax[2,2].fill_between(scat[:,6], u_obs[:,2]-np.std(u_obs[:,2]), u_obs[:,2]+np.std(u_obs[:,2]),color="orange",alpha=0.2)

    ax[2,0].plot(scat[:,3],u_mod[:,0],color="blue")
    ax[2,1].plot(scat[:,5],u_mod[:,1],color="red")
    ax[2,2].plot(scat[:,6],u_mod[:,2],color="orange")
    
    ax[1,0].set_ylabel('BRF(Q)', fontsize='x-large')
    ax[1,0].yaxis.set_label_coords(-.4, .5)
    
    ax[1,0].plot(scat[:,3],q_obs[:,0],linestyle='dashed',color="blue",label="470nm")
    ax[1,0].fill_between(scat[:,3], q_obs[:,0]-np.std(q_obs[:,0]), q_obs[:,0]+np.std(q_obs[:,0]),color="blue",alpha=0.2)
    #ax[1,0].fill_between(scat[3:5,3], q_obs[3:5,0]-np.std(q_obs[:,0]), q_obs[3:5,0]+np.std(q_obs[:,0]),facecolor="blue",alpha=0.2)

    ax[1,1].plot(scat[:,5],q_obs[:,1],linestyle='dashed',color="red",label="660nm")
    ax[1,1].fill_between(scat[:,5], q_obs[:,1]-0.005, q_obs[:,1]+0.005,color="red",alpha=0.2)
    #ax[1,1].fill_between(scat[3:5,3], q_obs[3:5,1]-np.std(q_obs[:,1]), q_obs[3:5,1]+np.std(q_obs[:,1]),facecolor="red",alpha=0.2)

    ax[1,2].plot(scat[:,6],q_obs[:,2],linestyle='dashed',color="orange",label="865nm")
    ax[1,2].fill_between(scat[:,6], q_obs[:,2]-np.std(q_obs[:,2]), q_obs[:,2]+np.std(q_obs[:,2]),color="orange",alpha=0.2)
    
    ax[1,0].plot(scat[:,3],q_mod[:,0],color="blue")
    ax[1,1].plot(scat[:,5],q_mod[:,1],color="red")
    ax[1,2].plot(scat[:,6],q_mod[:,2],color="orange")
    ax[2,1].set_xlabel(u"\u03A9 [\u00b0]",fontsize='xx-large')   
    
### END MAIN FUNCTION
if __name__ == '__main__':
     main() 
