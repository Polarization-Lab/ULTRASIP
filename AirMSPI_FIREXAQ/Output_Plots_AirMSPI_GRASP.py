"""
Output_Plot_AirMSPI_GRASP.py
INPUT: GRASP Output .txt File
OUTPUT: Figures of plots of GRASP Retrival Results

This is a Python 3.9.13 code to read in the output text file from the 
Generalized Retrieval of Atmosphere and Surface Properties retrieval on the 
given set of measurements. 

Code Sections: 
1. Load in Data
2. Plot Data 
a. I, Q, U vs Scattering Angle
b. I,Q,U vs View Angle
c. Retreival Results vs Measurements
d. Difference in Retrieved I, Q,U and Measured I,Q,U vs Scattering Angle
e. Difference in Retrieved I, Q,U and Measured I,Q,U vs View Angle

    
More info on the GRASP algorithm can be found at grasp-open.com
and more info on this complementary algoritm can be found in DeLeon et. al. (YYYY)

Creation Date: 2022-08-05
Last Modified: 2022-12-01

by Michael J. Garay and Clarissa M. DeLeon
(Michael.J.Garay@jpl.nasa.gov, cdeleon@arizona.edu)
"""
#_______________Import Packages_________________#

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

#_______________Start the main code____________#

def main():  

# Set the paths
# NOTE: outpath is the location of the GRASP output files
#       figpath is where the image output should be stored
#Work computer
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Setting_Files"
    figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Figures"

#Home computer

#_______________Load in Output File(s)___________________#
# Change directory to the basepath
    os.chdir(outpath)

# Find all output files with the specified file name 
    file_list = glob.glob('*GRASP*.txt')  
    num_files = len(file_list)   
    print("FOUND FILES:",num_files)

#_______________Read in the Data___________________#
# Set the length of a sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)
    num_step = 5
    
#num_int = total number of radiometric channels
#num_pol = total number of polarimetric channels

    num_int = 8 
    num_pol = 3
    
# Create arrays to store data
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
    
# READ DATA
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
    
        if(data_count==94):  # Sphericity
            words = line.split()
            per_sphere = float(words[1])
            
        if(data_count==100):  # AOD
            words = line.split()
            wave[0] = float(words[0])
            aod[0] = float(words[1])
            
        if(data_count==101):
            words = line.split()
            wave[1] = float(words[0])
            aod[1] = float(words[1])
        
        if(data_count==102):
            words = line.split()
            wave[2] = float(words[0])
            aod[2] = float(words[1])
            
        if(data_count==104):  # SSA
            words = line.split()
            ssa[0] = float(words[1])
            
        if(data_count==105):
            words = line.split()
            ssa[1] = float(words[1])
        
        if(data_count==106):
            words = line.split()
            ssa[2] = float(words[1])
            
        if(data_count==112):  # N_r
            words = line.split()
            nr[0] = float(words[1])
            
        if(data_count==113):
            words = line.split()
            nr[1] = float(words[1])
        
        if(data_count==114):
            words = line.split()
            nr[2] = float(words[1])
            
        if(data_count==116):  # N_i
            words = line.split()
            ni[0] = float(words[1])
            
        if(data_count==117):
            words = line.split()
            ni[1] = float(words[1])
        
        if(data_count==118):
            words = line.split()
            ni[2] = float(words[1])
            
        if(data_count==125):  # Fitting results
            words = line.split()
            vza[0,0] = float(words[2])
            scat[0,0]=float(words[4])
            i_obs[0,0] = float(words[5])
            i_mod[0,0] = float(words[6])
            
        if(data_count==126):  # Fitting results
            words = line.split()
            vza[1,0] = float(words[2])
            scat[1,0]=float(words[4])
            i_obs[1,0] = float(words[5])
            i_mod[1,0] = float(words[6])
            
        if(data_count==127):  # Fitting results
            words = line.split()
            vza[2,0] = float(words[2])
            scat[2,0]=float(words[4])
            i_obs[2,0] = float(words[5])
            i_mod[2,0] = float(words[6])
            
        if(data_count==128):  # Fitting results
            words = line.split()
            vza[3,0] = float(words[2])
            scat[3,0]=float(words[4])
            i_obs[3,0] = float(words[5])
            i_mod[3,0] = float(words[6])
            
        if(data_count==129):  # Fitting results
            words = line.split()
            vza[4,0] = float(words[2])
            scat[4,0]=float(words[4])
            i_obs[4,0] = float(words[5])
            i_mod[4,0] = float(words[6])
            
        if(data_count==146):  # Fitting results
            words = line.split()
            vza[0,1] = float(words[2])
            scat[0,1]=float(words[4])
            i_obs[0,1] = float(words[5])
            i_mod[0,1] = float(words[6])
            
        if(data_count==147):  # Fitting results
            words = line.split()
            vza[1,1] = float(words[2])
            scat[1,1]=float(words[4])
            i_obs[1,1] = float(words[5])
            i_mod[1,1] = float(words[6])
            
        if(data_count==148):  # Fitting results
            words = line.split()
            vza[2,1] = float(words[2])
            scat[2,1]=float(words[4])
            i_obs[2,1] = float(words[5])
            i_mod[2,1] = float(words[6])
            
        if(data_count==149):  # Fitting results
            words = line.split()
            vza[3,1] = float(words[2])
            scat[3,1]=float(words[4])
            i_obs[3,1] = float(words[5])
            i_mod[3,1] = float(words[6])
            
        if(data_count==150):  # Fitting results
            words = line.split()
            vza[4,1] = float(words[2])
            scat[4,1]=float(words[4])
            i_obs[4,1] = float(words[5])
            i_mod[4,1] = float(words[6])
            
        if(data_count==167):  # Fitting results
            words = line.split()
            vza[0,2] = float(words[2])
            scat[0,2]=float(words[4])
            i_obs[0,2] = float(words[5])
            i_mod[0,2] = float(words[6])
            
        if(data_count==168):  # Fitting results
            words = line.split()
            vza[1,2] = float(words[2])
            scat[1,2]=float(words[4])
            i_obs[1,2] = float(words[5])
            i_mod[1,2] = float(words[6])
            
        if(data_count==169):  # Fitting results
            words = line.split()
            vza[2,2] = float(words[2])
            scat[2,2]=float(words[4])
            i_obs[2,2] = float(words[5])
            i_mod[2,2] = float(words[6])
            
        if(data_count==170):  # Fitting results
            words = line.split()
            vza[3,2] = float(words[2])
            scat[3,2]=float(words[4])
            i_obs[3,2] = float(words[5])
            i_mod[3,2] = float(words[6])
            
        if(data_count==171):  # Fitting results
            words = line.split()
            vza[4,2] = float(words[2])
            scat[4,2]=float(words[4])
            i_obs[4,2] = float(words[5])
            i_mod[4,2] = float(words[6])
            
        if(data_count==131):  # Q
            words = line.split()
            q_obs[0,0] = float(words[5])
            q_mod[0,0] = float(words[6])
            
        if(data_count==132):  # Q
            words = line.split()
            q_obs[1,0] = float(words[5])
            q_mod[1,0] = float(words[6])
            
        if(data_count==133):  # Q
            words = line.split()
            q_obs[2,0] = float(words[5])
            q_mod[2,0] = float(words[6])
            
        if(data_count==134):  # Q
            words = line.split()
            q_obs[3,0] = float(words[5])
            q_mod[3,0] = float(words[6])
            
        if(data_count==135):  # Q
            words = line.split()
            q_obs[4,0] = float(words[5])
            q_mod[4,0] = float(words[6])
            
        if(data_count==152):  # Q
            words = line.split()
            q_obs[0,1] = float(words[5])
            q_mod[0,1] = float(words[6])
            
        if(data_count==153):  # Q
            words = line.split()
            q_obs[1,1] = float(words[5])
            q_mod[1,1] = float(words[6])
            
        if(data_count==154):  # Q
            words = line.split()
            q_obs[2,1] = float(words[5])
            q_mod[2,1] = float(words[6])
            
        if(data_count==155):  # Q
            words = line.split()
            q_obs[3,1] = float(words[5])
            q_mod[3,1] = float(words[6])
            
        if(data_count==156):  # Q
            words = line.split()
            q_obs[4,1] = float(words[5])
            q_mod[4,1] = float(words[6])
            
        if(data_count==173):  # Q
            words = line.split()
            q_obs[0,2] = float(words[5])
            q_mod[0,2] = float(words[6])
            
        if(data_count==174):  # Q
            words = line.split()
            q_obs[1,2] = float(words[5])
            q_mod[1,2] = float(words[6])
            
        if(data_count==175):  # Q
            words = line.split()
            q_obs[2,2] = float(words[5])
            q_mod[2,2] = float(words[6])
            
        if(data_count==176):  # Q
            words = line.split()
            q_obs[3,2] = float(words[5])
            q_mod[3,2] = float(words[6])
            
        if(data_count==177):  # Q
            words = line.split()
            q_obs[4,2] = float(words[5])
            q_mod[4,2] = float(words[6])
            
        if(data_count==137):  # U
            words = line.split()
            u_obs[0,0] = float(words[5])
            u_mod[0,0] = float(words[6])
            
        if(data_count==138):  # U
            words = line.split()
            u_obs[1,0] = float(words[5])
            u_mod[1,0] = float(words[6])
            
        if(data_count==139):  # U
            words = line.split()
            u_obs[2,0] = float(words[5])
            u_mod[2,0] = float(words[6])
            
        if(data_count==140):  # U
            words = line.split()
            u_obs[3,0] = float(words[5])
            u_mod[3,0] = float(words[6])
            
        if(data_count==141):  # U
            words = line.split()
            u_obs[4,0] = float(words[5])
            u_mod[4,0] = float(words[6])
            
        if(data_count==158):  # U
            words = line.split()
            u_obs[0,1] = float(words[5])
            u_mod[0,1] = float(words[6])
            
        if(data_count==159):  # U
            words = line.split()
            u_obs[1,1] = float(words[5])
            u_mod[1,1] = float(words[6])
            
        if(data_count==160):  # U
            words = line.split()
            u_obs[2,1] = float(words[5])
            u_mod[2,1] = float(words[6])
            
        if(data_count==161):  # U
            words = line.split()
            u_obs[3,1] = float(words[5])
            u_mod[3,1] = float(words[6])
            
        if(data_count==162):  # U
            words = line.split()
            u_obs[4,1] = float(words[5])
            u_mod[4,1] = float(words[6])
            
        if(data_count==179):  # U
            words = line.split()
            u_obs[0,2] = float(words[5])
            u_mod[0,2] = float(words[6])
            
        if(data_count==180):  # U
            words = line.split()
            u_obs[1,2] = float(words[5])
            u_mod[1,2] = float(words[6])
            
        if(data_count==181):  # U
            words = line.split()
            u_obs[2,2] = float(words[5])
            u_mod[2,2] = float(words[6])
            
        if(data_count==182):  # U
            words = line.split()
            u_obs[3,2] = float(words[5])
            u_mod[3,2] = float(words[6])
            
        if(data_count==183):  # U
            words = line.split()
            u_obs[4,2] = float(words[5])
            u_mod[4,2] = float(words[6])
            
        data_count = data_count+1

# Close the input file

    inputFile.close()    
    
#_______________Section 2: Plot Data___________________#
# Change to the figure directory
    os.chdir(figpath)

## FIRST PLOT - INTENSITY VS. SCATTERING ANGLE
# Upper Left - Intensity       Upper Right - Q
# Lower Left - U               Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)
        
# Polarized Bands - I

    print(max(i_obs[:,0]))

    ax1.plot(scat[:,0],i_obs[:,0],marker='o',color="blue",label="470nm")
    ax1.plot(scat[:,1],i_obs[:,1],marker='o',color="red",label="660nm")
    ax1.plot(scat[:,2],i_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax1.plot(scat[:,0],i_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax1.plot(scat[:,1],i_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax1.plot(scat[:,2],i_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax1.set_xlim(60,180)
    ax1.set_xticks(np.arange(60,190,30))
    ax1.set_xlabel("Scattering Angle (Deg)")
    
    ax1.set_ylim(0.0,0.4)
    ax1.set_yticks(np.arange(0.0,0.5,0.10))
    ax1.set_ylabel('Equivalent Reflectance')
    
    ax1.legend(loc=1)  # Upper right
    
# Polarized Bands - Q

    ax2.plot(scat[:,0],q_obs[:,0],marker='o',color="blue",label="470nm")
    ax2.plot(scat[:,1],q_obs[:,1],marker='o',color="red",label="660nm")
    ax2.plot(scat[:,2],q_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax2.plot(scat[:,0],q_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax2.plot(scat[:,1],q_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax2.plot(scat[:,2],q_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax2.set_xlim(60,180)
    ax2.set_xticks(np.arange(60,190,30))
    ax2.set_xlabel("Scattering Angle (Deg)")
    
    ax2.set_ylim(-0.5,0.5)
    ax2.set_yticks(np.arange(-0.5,0.75,0.25))
    ax2.set_ylabel('q (Q/I)')
    
    ax2.legend(loc=1)  # Upper right

# Polarized Bands - U

    ax3.plot(scat[:,0],u_obs[:,0],marker='o',color="blue",label="470nm")
    ax3.plot(scat[:,1],u_obs[:,1],marker='o',color="red",label="660nm")
    ax3.plot(scat[:,2],u_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax3.plot(scat[:,0],u_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax3.plot(scat[:,1],u_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax3.plot(scat[:,2],u_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax3.set_xlim(60,180)
    ax3.set_xticks(np.arange(60,190,30))
    ax3.set_xlabel("Scattering Angle (Deg)")
    
    ax3.set_ylim(-0.5,0.5)
    ax3.set_yticks(np.arange(-0.5,0.75,0.25))
    ax3.set_ylabel('u (U/I)')
    
    ax3.legend(loc=1)  # Upper right
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "ADDITIONAL INFORMATION"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (470 nm): {:6.3f}'.format(aod[0]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (470 nm): {:6.3f}'.format(ssa[0]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '      N_r (470) nm): {:7.4f}'.format(nr[0]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (470 nm): {:7.4f}'.format(ni[0]) 
    ax4.text(80,0.13,out_text)
    
    ax4.set_xlim(60,180)
    ax4.set_xticks(np.arange(60,190,30))
    
    ax4.set_ylim(0.0,0.4)
    ax4.set_yticks(np.arange(0.0,0.5,0.10))
    
    ax4.axis('off')
    
    plt.tight_layout()

# Get the software version number to help track issues

    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0] 

# Generate the output file name
    
    hold = inputName.split(".")
    out_base = hold[0]
    outfile = out_base+'_VIS_v'+vers+'_01.png'
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 

## SECOND PLOT - INTENSITY VS. VIEW ANGLE
# Upper Left - Intensity       Upper Right - Q
# Lower Left - U               Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# NOTE: We make the first two view zenith angles negative to plot the results
#       in "camera" order (first observation to last observation)

    vza_plot = np.copy(vza)
    vza_plot[0,:] = -1.0*vza[0,:]
    vza_plot[1,:] = -1.0*vza[1,:]
    
# Polarized Bands - I

    ax1.plot(vza_plot[:,0],i_obs[:,0],marker='o',color="blue",label="470nm")
    ax1.plot(vza_plot[:,1],i_obs[:,1],marker='o',color="red",label="660nm")
    ax1.plot(vza_plot[:,2],i_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax1.plot(vza_plot[:,0],i_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax1.plot(vza_plot[:,1],i_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax1.plot(vza_plot[:,2],i_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax1.set_xlim(-80,80)
    ax1.set_xticks(np.arange(-80,90,20))
    ax1.set_xlabel("View Angle (Deg)")
    
    ax1.set_ylim(0.0,0.4)
    ax1.set_yticks(np.arange(0.0,0.5,0.10))
    ax1.set_ylabel('Equivalent Reflectance')
    
    ax1.legend(loc=1)  # Upper right
    
# Polarized Bands - Q

    ax2.plot(vza_plot[:,0],q_obs[:,0],marker='o',color="blue",label="470nm")
    ax2.plot(vza_plot[:,1],q_obs[:,1],marker='o',color="red",label="660nm")
    ax2.plot(vza_plot[:,2],q_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax2.plot(vza_plot[:,0],q_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax2.plot(vza_plot[:,1],q_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax2.plot(vza_plot[:,2],q_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax2.set_xlim(-80,80)
    ax2.set_xticks(np.arange(-80,90,20))
    ax2.set_xlabel("View Angle (Deg)")
    
    ax2.set_ylim(-0.5,0.5)
    ax2.set_yticks(np.arange(-0.5,0.75,0.25))
    ax2.set_ylabel('q (Q/I)')
        
    ax2.legend(loc=1)  # Upper right
    
# Polarized Bands - U

    ax3.plot(vza_plot[:,0],u_obs[:,0],marker='o',color="blue",label="470nm")
    ax3.plot(vza_plot[:,1],u_obs[:,1],marker='o',color="red",label="660nm")
    ax3.plot(vza_plot[:,2],u_obs[:,2],marker='o',color="magenta",label="865nm")
    
    ax3.plot(vza_plot[:,0],u_mod[:,0],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax3.plot(vza_plot[:,1],u_mod[:,1],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax3.plot(vza_plot[:,2],u_mod[:,2],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax3.set_xlim(-80,80)
    ax3.set_xticks(np.arange(-80,90,20))
    ax3.set_xlabel("View Angle (Deg)")
    
    ax3.set_ylim(-0.5,0.5)
    ax3.set_yticks(np.arange(-0.5,0.75,0.25))
    ax3.set_ylabel('u (U/I)')
        
    ax3.legend(loc=1)  # Upper right
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (470 nm): {:6.3f}'.format(aod[0]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (470 nm): {:6.3f}'.format(ssa[0]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '      N_r (470) nm): {:7.4f}'.format(nr[0]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (470 nm): {:7.4f}'.format(ni[0]) 
    ax4.text(80,0.13,out_text)
    
    ax4.set_xlim(60,180)
    ax4.set_xticks(np.arange(60,190,30))
    
    ax4.set_ylim(0.0,0.4)
    ax4.set_yticks(np.arange(0.0,0.5,0.10))
    
    ax4.axis('off')
    
    plt.tight_layout()

# Get the software version number to help track issues

    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0] 

# Generate the output file name
    
    hold = inputName.split(".")
    out_base = hold[0]
    outfile = out_base+'_VIS_v'+vers+'_02.png'
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 
    
# Show the plot
    
    plt.show() 

    plt.close()
    
## THIRD PLOT - MODEL VS. OBS
# Upper Left - Intensity       Upper Right - Q
# Lower Left - U               Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# Polarized Bands - I

    ax1.scatter(i_obs[:,0],i_mod[:,0],marker='o',color="blue",label="470nm")
    ax1.scatter(i_obs[:,1],i_mod[:,1],marker='o',color="red",label="660nm")
    ax1.scatter(i_obs[:,2],i_mod[:,2],marker='o',color="magenta",label="865nm")
 
    ax1.set_xlim(0.0,0.4)
    ax1.set_xticks(np.arange(0.0,0.5,0.10))
    ax1.set_xlabel('Equivalent Reflectance (AirMSPI)')
    
    ax1.set_ylim(0.0,0.4)
    ax1.set_yticks(np.arange(0.0,0.5,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP)')
    
    ax1.plot([0,0.5],[0,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax1.legend(loc=2)  # Upper left
    
# Polarized Bands - Q

    ax2.scatter(q_obs[:,0],q_mod[:,0],marker='o',color="blue",label="470nm")
    ax2.scatter(q_obs[:,1],q_mod[:,1],marker='o',color="red",label="660nm")
    ax2.scatter(q_obs[:,2],q_mod[:,2],marker='o',color="magenta",label="865nm")
 
    ax2.set_xlim(-0.5,0.5)
    ax2.set_xticks(np.arange(-0.5,0.75,0.25))
    ax2.set_xlabel('q (Q/I) (AirMSPI)')
    
    ax2.set_ylim(-0.5,0.5)
    ax2.set_yticks(np.arange(-0.5,0.75,0.25))
    ax2.set_ylabel('q (Q/I) (GRASP)')
    
    ax2.plot([-0.5,0.5],[-0.5,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax2.legend(loc=2)  # Upper left
    
# Polarized Bands - U

    ax3.scatter(u_obs[:,0],u_mod[:,0],marker='o',color="blue",label="470nm")
    ax3.scatter(u_obs[:,1],u_mod[:,1],marker='o',color="red",label="660nm")
    ax3.scatter(u_obs[:,2],u_mod[:,2],marker='o',color="magenta",label="865nm")
 
    ax3.set_xlim(-0.5,0.5)
    ax3.set_xticks(np.arange(-0.5,0.75,0.25))
    ax3.set_xlabel('u (U/I) (AirMSPI)')
    
    ax3.set_ylim(-0.5,0.5)
    ax3.set_yticks(np.arange(-0.5,0.75,0.25))
    ax3.set_ylabel('u (U/I) (GRASP)')
    
    ax3.plot([-0.5,0.5],[-0.5,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax3.legend(loc=2)  # Upper left
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (470 nm): {:6.3f}'.format(aod[0]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (470 nm): {:6.3f}'.format(ssa[0]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '      N_r (470) nm): {:7.4f}'.format(nr[0]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (470 nm): {:7.4f}'.format(ni[0]) 
    ax4.text(80,0.13,out_text)
    
    ax4.set_xlim(60,180)
    ax4.set_xticks(np.arange(60,190,30))
    
    ax4.set_ylim(0.0,0.4)
    ax4.set_yticks(np.arange(0.0,0.5,0.10))
    
    ax4.axis('off')
    
    plt.tight_layout()

# Get the software version number to help track issues

    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0] 

# Generate the output file name
    
    hold = inputName.split(".")
    out_base = hold[0]
    outfile = out_base+'_VIS_v'+vers+'_03.png'
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 
    
# Show the plot
    
    plt.show() 
    
## FOURTH PLOT - DELTA INTENSITY VS. SCATTERING ANGLE
# Upper Left - Intensity       Upper Right - Q
# Lower Left - U               Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# Note: We calculate delta obs as model minus observation

    delta_i = i_mod - i_obs
    delta_q = q_mod - q_obs
    delta_u = u_mod - u_obs
        
# Polarized Bands - I

    ax1.scatter(scat[:,0],delta_i[:,0],marker='o',color="blue",label="470nm")
    ax1.scatter(scat[:,1],delta_i[:,1],marker='o',color="red",label="660nm")
    ax1.scatter(scat[:,2],delta_i[:,2],marker='o',color="magenta",label="865nm")
    
    ax1.set_xlim(60,180)
    ax1.set_xticks(np.arange(60,190,30))
    ax1.set_xlabel("Scattering Angle (Deg)")
    
    ax1.set_ylim(-0.2,0.2)
    ax1.set_yticks(np.arange(-0.2,0.3,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax1.legend(loc=1)  # Upper right
    
    ax1.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands - Q

    ax2.scatter(scat[:,0],delta_q[:,0],marker='o',color="blue",label="470nm")
    ax2.scatter(scat[:,1],delta_q[:,1],marker='o',color="red",label="660nm")
    ax2.scatter(scat[:,2],delta_q[:,2],marker='o',color="magenta",label="865nm")
    
    ax2.set_xlim(60,180)
    ax2.set_xticks(np.arange(60,190,30))
    ax2.set_xlabel("Scattering Angle (Deg)")
    
    ax2.set_ylim(-0.3,0.3)
    ax2.set_yticks(np.arange(-0.3,0.4,0.10))
    ax2.set_ylabel('q (Q/I) (GRASP-AirMSPI)')
    
    ax2.legend(loc=1)  # Upper right
    
    ax2.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands - U

    ax3.scatter(scat[:,0],delta_u[:,0],marker='o',color="blue",label="470nm")
    ax3.scatter(scat[:,1],delta_u[:,1],marker='o',color="red",label="660nm")
    ax3.scatter(scat[:,2],delta_u[:,2],marker='o',color="magenta",label="865nm")
    
    ax3.set_xlim(60,180)
    ax3.set_xticks(np.arange(60,190,30))
    ax3.set_xlabel("Scattering Angle (Deg)")
    
    ax3.set_ylim(-0.3,0.3)
    ax3.set_yticks(np.arange(-0.3,0.4,0.10))
    ax3.set_ylabel('u (U/I) (GRASP-AirMSPI)')
    
    ax3.legend(loc=0)  # Upper left
    
    ax3.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (470 nm): {:6.3f}'.format(aod[0]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (470 nm): {:6.3f}'.format(ssa[0]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '      N_r (470) nm): {:7.4f}'.format(nr[0]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (470 nm): {:7.4f}'.format(ni[0]) 
    ax4.text(80,0.13,out_text)
    
    ax4.set_xlim(60,180)
    ax4.set_xticks(np.arange(60,190,30))
    
    ax4.set_ylim(0.0,0.4)
    ax4.set_yticks(np.arange(0.0,0.5,0.10))
    
    ax4.axis('off')
    
    plt.tight_layout()

# Get the software version number to help track issues

    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0] 

# Generate the output file name
    
    hold = inputName.split(".")
    out_base = hold[0]
    outfile = out_base+'_VIS_v'+vers+'_04.png'
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 
    
# Show the plot
    
    plt.show() 
    
## FIFTH PLOT - DELTA INTENSITY VS. VIEW ANGLE
# Upper Left - Intensity       Upper Right - Q
# Lower Left - U               Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# Polarized Bands - I

    ax1.scatter(vza_plot[:,0],delta_i[:,0],marker='o',color="blue",label="470nm")
    ax1.scatter(vza_plot[:,1],delta_i[:,1],marker='o',color="red",label="660nm")
    ax1.scatter(vza_plot[:,2],delta_i[:,2],marker='o',color="magenta",label="865nm")
    
    ax1.set_xlim(-80,80)
    ax1.set_xticks(np.arange(-80,90,20))
    ax1.set_xlabel("View Angle (Deg)")
    
    ax1.set_ylim(-0.2,0.2)
    ax1.set_yticks(np.arange(-0.2,0.3,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax1.legend(loc=1)  # Upper right
    
    ax1.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands - Q

    ax2.scatter(vza_plot[:,0],delta_q[:,0],marker='o',color="blue",label="470nm")
    ax2.scatter(vza_plot[:,1],delta_q[:,1],marker='o',color="red",label="660nm")
    ax2.scatter(vza_plot[:,2],delta_q[:,2],marker='o',color="magenta",label="865nm")
    
    ax2.set_xlim(-80,80)
    ax2.set_xticks(np.arange(-80,90,20))
    ax2.set_xlabel("View Angle (Deg)")
    
    ax2.set_ylim(-0.3,0.3)
    ax2.set_yticks(np.arange(-0.3,0.4,0.10))
    ax2.set_ylabel('q (Q/I) (GRASP-AirMSPI)')
    
    ax2.legend(loc=1)  # Upper right
    
    ax2.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands - U

    ax3.scatter(vza_plot[:,0],delta_u[:,0],marker='o',color="blue",label="470nm")
    ax3.scatter(vza_plot[:,1],delta_u[:,1],marker='o',color="red",label="660nm")
    ax3.scatter(vza_plot[:,2],delta_u[:,2],marker='o',color="magenta",label="865nm")
    
    ax3.set_xlim(-80,80)
    ax3.set_xticks(np.arange(-80,90,20))
    ax3.set_xlabel("View Angle (Deg)")
    
    ax3.set_ylim(-0.3,0.3)
    ax3.set_yticks(np.arange(-0.3,0.4,0.10))
    ax3.set_ylabel('u (U/I) (GRASP-AirMSPI)')
    
    ax3.legend(loc=0)  # Upper left
    
    ax3.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (470 nm): {:6.3f}'.format(aod[0]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (470 nm): {:6.3f}'.format(ssa[0]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '      N_r (470) nm): {:7.4f}'.format(nr[0]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (470 nm): {:7.4f}'.format(ni[0]) 
    ax4.text(80,0.13,out_text)
    
    ax4.set_xlim(60,180)
    ax4.set_xticks(np.arange(60,190,30))
    
    ax4.set_ylim(0.0,0.4)
    ax4.set_yticks(np.arange(0.0,0.5,0.10))
    
    ax4.axis('off')
    
    plt.tight_layout()
    
# Get the software version number to help track issues

    hold = os.path.basename(__file__)
    words = hold.split('_')
    temp = words[len(words)-1]  # Choose the last element
    hold = temp.split('.')
    vers = hold[0] 

# Generate the output file name
    
    hold = inputName.split(".")
    out_base = hold[0]
    outfile = out_base+'_VIS_v'+vers+'_05.png'
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 
    
# Show the plot
    
    plt.show() 
    
    plt.close()
    
# Tell user completion was successful

    print("\nSuccessful Completion\n")

### END MAIN FUNCTION


if __name__ == '__main__':
    main() 