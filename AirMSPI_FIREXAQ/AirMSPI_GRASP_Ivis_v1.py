# GRASP_Prescott_output_I_vis_14.py
#
# This is a Python 3.9.13 code to read the output of the GRASP retrieval code
# for AirMSPI observations over Prescott, AZ (intensity only)
# and plot the results.
#
# Creation Date: 2022-08-08
# Last Modified: 2022-08-08
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

# Import packages

import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def main():  # Main code

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

    file_list = glob.glob('*I_GRASP*.txt')
    
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

            if((words[0]=="Wavelength")&(aod_count<1)):
                aod_start = data_count
                
# Read the AOD

            if((data_count>aod_start)&(aod_count<num_int)):
                wave[aod_count] = float(words[0])
                aod[aod_count] = float(words[1])
                aod_count = aod_count+1
            
# Check for "WAVELENGTH" - SECOND IS FOR SSA

            if((words[0]=="Wavelength")&(aod_count>0)):
                ssa_start = data_count
                
# Read the SSA

            if((data_count>ssa_start)&(ssa_count<num_int)):
                ssa[ssa_count] = float(words[1])
                ssa_count = ssa_count+1
                
# Check for "WAVELENGTH" - THIRD IS FOR AAOD

            if((words[0]=="Wavelength")&(ssa_count>0)):
                aaod_start = data_count
                
# Read the AAOD

            if((data_count>aaod_start)&(aaod_count<num_int)):
                aaod[aaod_count] = float(words[1])
                aaod_count = aaod_count+1
                
# Check for "WAVELENGTH" - FOURTH IS FOR THE REAL PART OF THE REFRACTIVE INDEX

            if((words[0]=="Wavelength")&(aaod_count>0)):
                nr_start = data_count
                
# Read the real part of the refractive index

            if((data_count>nr_start)&(nr_count<num_int)):
                nr[nr_count] = float(words[1])
                nr_count = nr_count+1
                
# Check for "WAVELENGTH" - FIFTH IS FOR THE IMAGINARY PART OF THE REFRACTIVE INDEX

            if((words[0]=="Wavelength")&(nr_count>0)):
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

### PLOT THE DATA

# Change to the figure directory

    os.chdir(figpath)

## FIRST PLOT - INTENSITY VS. SCATTERING ANGLE
# Upper Left - UV bands       Upper Right - Polarized bands
# Lower Left - Non-pol bands  Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)
        
# UV Bands

    ax1.plot(scat[:,0],i_obs[:,0],marker='o',color="indigo",label="355nm")
    ax1.plot(scat[:,1],i_obs[:,1],marker='o',color="purple",label="380nm")
    
    ax1.plot(scat[:,0],i_mod[:,0],marker='D',color="indigo",fillstyle='none',linestyle='dashed')
    ax1.plot(scat[:,1],i_mod[:,1],marker='D',color="purple",fillstyle='none',linestyle='dashed')
    
    ax1.set_xlim(60,180)
    ax1.set_xticks(np.arange(60,190,30))
    ax1.set_xlabel("Scattering Angle (Deg)")
    
    ax1.set_ylim(0.0,0.5)
    ax1.set_yticks(np.arange(0.0,0.6,0.10))
    ax1.set_ylabel('Equivalent Reflectance')
    
    ax1.legend(loc=1)  # Upper right
    
# Polarized Bands

    ax2.plot(scat[:,3],i_obs[:,3],marker='o',color="blue",label="470nm")
    ax2.plot(scat[:,5],i_obs[:,5],marker='o',color="red",label="660nm")
    ax2.plot(scat[:,6],i_obs[:,6],marker='o',color="magenta",label="865nm")
    
    ax2.plot(scat[:,3],i_mod[:,3],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax2.plot(scat[:,5],i_mod[:,5],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax2.plot(scat[:,6],i_mod[:,6],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax2.set_xlim(60,180)
    ax2.set_xticks(np.arange(60,190,30))
    ax2.set_xlabel("Scattering Angle (Deg)")
    
    ax2.set_ylim(0.0,0.4)
    ax2.set_yticks(np.arange(0.0,0.5,0.10))
    ax2.set_ylabel('Equivalent Reflectance')
    
    ax2.legend(loc=1)  # Upper right

# Additional Non-polarimetric Bands

    ax3.plot(scat[:,2],i_obs[:,2],marker='o',color="navy",label="445nm")
    ax3.plot(scat[:,4],i_obs[:,4],marker='o',color="lime",label="555nm")
    
    ax3.plot(scat[:,2],i_mod[:,2],marker='D',color="navy",fillstyle='none',linestyle='dashed')
    ax3.plot(scat[:,4],i_mod[:,4],marker='D',color="lime",fillstyle='none',linestyle='dashed')

    ax3.set_xlim(60,180)
    ax3.set_xticks(np.arange(60,190,30))
    ax3.set_xlabel("Scattering Angle (Deg)")
    
    ax3.set_ylim(0.0,0.4)
    ax3.set_yticks(np.arange(0.0,0.5,0.10))
    ax3.set_ylabel('Equivalent Reflectance')
    
    ax3.legend(loc=1)  # Upper right
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (555 nm): {:6.3f}'.format(aod[4]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (555 nm): {:6.3f}'.format(ssa[4]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '       N_r (555 nm): {:7.4f}'.format(nr[4]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (555 nm): {:7.4f}'.format(ni[4]) 
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
    
# Show the plot
    
    plt.show() 
    
    plt.close()
    
## SECOND PLOT - INTENSITY VS. VIEW ANGLE
# Upper Left - UV bands       Upper Right - Polarized bands
# Lower Left - Non-pol bands  Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# NOTE: We make the first two view zenith angles negative to plot the results
#       in "camera" order (first observation to last observation)

    vza_plot = np.copy(vza)
    vza_plot[0,:] = -1.0*vza[0,:]
    vza_plot[1,:] = -1.0*vza[1,:]

# UV Bands

    ax1.plot(vza_plot[:,0],i_obs[:,0],marker='o',color="indigo",label="355nm")
    ax1.plot(vza_plot[:,1],i_obs[:,1],marker='o',color="purple",label="380nm")
    
    ax1.plot(vza_plot[:,0],i_mod[:,0],marker='D',color="indigo",fillstyle='none',linestyle='dashed')
    ax1.plot(vza_plot[:,1],i_mod[:,1],marker='D',color="purple",fillstyle='none',linestyle='dashed')
    
    ax1.set_xlim(-80,80)
    ax1.set_xticks(np.arange(-80,90,20))
    ax1.set_xlabel("View Angle (Deg)")
    
    ax1.set_ylim(0.0,0.5)
    ax1.set_yticks(np.arange(0.0,0.6,0.10))
    ax1.set_ylabel('Equivalent Reflectance')
    
    ax1.legend(loc=1)  # Upper right
    
# Polarized Bands

    ax2.plot(vza_plot[:,3],i_obs[:,3],marker='o',color="blue",label="470nm")
    ax2.plot(vza_plot[:,5],i_obs[:,5],marker='o',color="red",label="660nm")
    ax2.plot(vza_plot[:,6],i_obs[:,6],marker='o',color="magenta",label="865nm")
    
    ax2.plot(vza_plot[:,3],i_mod[:,3],marker='D',color="blue",fillstyle='none',linestyle='dashed')
    ax2.plot(vza_plot[:,5],i_mod[:,5],marker='D',color="red",fillstyle='none',linestyle='dashed')
    ax2.plot(vza_plot[:,6],i_mod[:,6],marker='D',color="magenta",fillstyle='none',linestyle='dashed')
 
    ax2.set_xlim(-80,80)
    ax2.set_xticks(np.arange(-80,90,20))
    ax2.set_xlabel("View Angle (Deg)")
    
    ax2.set_ylim(0.0,0.4)
    ax2.set_yticks(np.arange(0.0,0.5,0.10))
    ax2.set_ylabel('Equivalent Reflectance')
    
    ax2.legend(loc=1)  # Upper right

# Additional Non-polarimetric Bands

    ax3.plot(vza_plot[:,2],i_obs[:,2],marker='o',color="navy",label="445nm")
    ax3.plot(vza_plot[:,4],i_obs[:,4],marker='o',color="lime",label="555nm")
    
    ax3.plot(vza_plot[:,2],i_mod[:,2],marker='D',color="navy",fillstyle='none',linestyle='dashed')
    ax3.plot(vza_plot[:,4],i_mod[:,4],marker='D',color="lime",fillstyle='none',linestyle='dashed')

    ax3.set_xlim(-80,80)
    ax3.set_xticks(np.arange(-80,90,20))
    ax3.set_xlabel("View Angle (Deg)")
    
    ax3.set_ylim(0.0,0.4)
    ax3.set_yticks(np.arange(0.0,0.5,0.10))
    ax3.set_ylabel('Equivalent Reflectance')
    
    ax3.legend(loc=1)  # Upper right
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (555 nm): {:6.3f}'.format(aod[4]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (555 nm): {:6.3f}'.format(ssa[4]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '       N_r (555 nm): {:7.4f}'.format(nr[4]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (555 nm): {:7.4f}'.format(ni[4]) 
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
# Upper Left - UV bands       Upper Right - Polarized bands
# Lower Left - Non-pol bands  Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# UV Bands

    ax1.scatter(i_obs[:,0],i_mod[:,0],marker='o',color="indigo",label="355nm")
    ax1.scatter(i_obs[:,1],i_mod[:,1],marker='o',color="purple",label="380nm")

    ax1.set_xlim(0.0,0.5)
    ax1.set_xticks(np.arange(0.0,0.6,0.10))
    ax1.set_xlabel('Equivalent Reflectance (Obs)')
    
    ax1.set_ylim(0.0,0.5)
    ax1.set_yticks(np.arange(0.0,0.6,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP)')
    
    ax1.plot([0,0.5],[0,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax1.legend(loc=2)  # Upper left
    
# Polarized Bands

    ax2.scatter(i_obs[:,3],i_mod[:,3],marker='o',color="blue",label="470nm")
    ax2.scatter(i_obs[:,5],i_mod[:,5],marker='o',color="red",label="660nm")
    ax2.scatter(i_obs[:,6],i_mod[:,6],marker='o',color="magenta",label="865nm")
 
    ax2.set_xlim(0.0,0.4)
    ax2.set_xticks(np.arange(0.0,0.5,0.10))
    ax2.set_xlabel('Equivalent Reflectance (AirMSPI)')
    
    ax2.set_ylim(0.0,0.4)
    ax2.set_yticks(np.arange(0.0,0.5,0.10))
    ax2.set_ylabel('Equivalent Reflectance (GRASP)')
    
    ax2.plot([0,0.5],[0,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax2.legend(loc=2)  # Upper left
    
# Additional Non-polarimetric Bands

    ax3.scatter(i_obs[:,2],i_mod[:,2],marker='o',color="navy",label="445nm")
    ax3.scatter(i_obs[:,4],i_mod[:,4],marker='o',color="lime",label="555nm")

    ax3.set_xlim(0.0,0.4)
    ax3.set_xticks(np.arange(0.0,0.5,0.10))
    ax3.set_xlabel('Equivalent Reflectance (AirMSPI)')
    
    ax3.set_ylim(0.0,0.4)
    ax3.set_yticks(np.arange(0.0,0.5,0.10))
    ax3.set_ylabel('Equivalent Reflectance (GRASP)')
    
    ax3.plot([0,0.5],[0,0.5],color="black",linewidth=1) # One-to-one Line
    
    ax3.legend(loc=2)  # Upper left
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (555 nm): {:6.3f}'.format(aod[4]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (555 nm): {:6.3f}'.format(ssa[4]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '       N_r (555 nm): {:7.4f}'.format(nr[4]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (555 nm): {:7.4f}'.format(ni[4]) 
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
    
    plt.close()
    
## FOURTH PLOT - DELTA INTENSITY VS. SCATTERING ANGLE
# Upper Left - UV bands       Upper Right - Polarized bands
# Lower Left - Non-pol bands  Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# Note: We calculate delta obs as model minus observation

    delta_obs = i_mod - i_obs
        
# UV Bands

    ax1.scatter(scat[:,0],delta_obs[:,0],marker='o',color="indigo",label="355nm")
    ax1.scatter(scat[:,1],delta_obs[:,1],marker='o',color="purple",label="380nm")
    
    ax1.set_xlim(60,180)
    ax1.set_xticks(np.arange(60,190,30))
    ax1.set_xlabel("Scattering Angle (Deg)")
    
    ax1.set_ylim(-0.2,0.2)
    ax1.set_yticks(np.arange(-0.2,0.3,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax1.legend(loc=1)  # Upper right
    
    ax1.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands

    ax2.scatter(scat[:,3],delta_obs[:,3],marker='o',color="blue",label="470nm")
    ax2.scatter(scat[:,5],delta_obs[:,5],marker='o',color="red",label="660nm")
    ax2.scatter(scat[:,6],delta_obs[:,6],marker='o',color="magenta",label="865nm")
    
    ax2.set_xlim(60,180)
    ax2.set_xticks(np.arange(60,190,30))
    ax2.set_xlabel("Scattering Angle (Deg)")
    
    ax2.set_ylim(-0.2,0.2)
    ax2.set_yticks(np.arange(-0.2,0.3,0.10))
    ax2.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax2.legend(loc=1)  # Upper right
    
    ax2.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Additional Non-polarimetric Bands

    ax3.scatter(scat[:,2],delta_obs[:,2],marker='o',color="navy",label="445nm")
    ax3.scatter(scat[:,4],delta_obs[:,4],marker='o',color="lime",label="555nm")
    
    ax3.set_xlim(60,180)
    ax3.set_xticks(np.arange(60,190,30))
    ax3.set_xlabel("Scattering Angle (Deg)")
    
    ax3.set_ylim(-0.2,0.2)
    ax3.set_yticks(np.arange(-0.2,0.3,0.10))
    ax3.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax3.legend(loc=1)  # Upper right
    
    ax3.plot([60,180],[0,0],color="black",linewidth=1) # Zero Line
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (555 nm): {:6.3f}'.format(aod[4]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (555 nm): {:6.3f}'.format(ssa[4]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '       N_r (555 nm): {:7.4f}'.format(nr[4]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (555 nm): {:7.4f}'.format(ni[4]) 
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
    
    plt.close()
    
## FIFTH PLOT - DELTA INTENSITY VS. VIEW ANGLE
# Upper Left - UV bands       Upper Right - Polarized bands
# Lower Left - Non-pol bands  Lower Right - Retrieval Info

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(9,6), 
        nrows=2, ncols=2, dpi=120)

# UV Bands

    ax1.scatter(vza_plot[:,0],delta_obs[:,0],marker='o',color="indigo",label="355nm")
    ax1.scatter(vza_plot[:,1],delta_obs[:,1],marker='o',color="purple",label="380nm")
    
    ax1.set_xlim(-80,80)
    ax1.set_xticks(np.arange(-80,90,20))
    ax1.set_xlabel("View Angle (Deg)")
    
    ax1.set_ylim(-0.2,0.2)
    ax1.set_yticks(np.arange(-0.2,0.3,0.10))
    ax1.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax1.legend(loc=1)  # Upper right
    
    ax1.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line
    
# Polarized Bands

    ax2.scatter(vza_plot[:,3],delta_obs[:,3],marker='o',color="blue",label="470nm")
    ax2.scatter(vza_plot[:,5],delta_obs[:,5],marker='o',color="red",label="660nm")
    ax2.scatter(vza_plot[:,6],delta_obs[:,6],marker='o',color="magenta",label="865nm")
    
    ax2.set_xlim(-80,80)
    ax2.set_xticks(np.arange(-80,90,20))
    ax2.set_xlabel("View Angle (Deg)")
    
    ax2.set_ylim(-0.2,0.2)
    ax2.set_yticks(np.arange(-0.2,0.3,0.10))
    ax2.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax2.legend(loc=1)  # Upper right
    
    ax2.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line

# Additional Non-polarimetric Bands

    ax3.scatter(vza_plot[:,2],delta_obs[:,2],marker='o',color="navy",label="445nm")
    ax3.scatter(vza_plot[:,4],delta_obs[:,4],marker='o',color="lime",label="555nm")
    
    ax3.set_xlim(-80,80)
    ax3.set_xticks(np.arange(-80,90,20))
    ax3.set_xlabel("View Angle (Deg)")
    
    ax3.set_ylim(-0.2,0.2)
    ax3.set_yticks(np.arange(-0.2,0.3,0.10))
    ax3.set_ylabel('Equivalent Reflectance (GRASP-AirMSPI)')
    
    ax3.legend(loc=1)  # Upper right
    
    ax3.plot([-80,80],[0,0],color="black",linewidth=1) # Zero Line
    
# Text information
# NOTE: I'm going to use the scattering angle coordinates to locate the text
    
    out_text = "Retrieved Microphysical Properties"
    ax4.text(80,0.30,out_text,fontweight='bold')
    
    out_text = '     AOD (555 nm): {:6.3f}'.format(aod[4]) 
    ax4.text(80,0.25,out_text)
    
    out_text = '      SSA (555 nm): {:6.3f}'.format(ssa[4]) 
    ax4.text(80,0.22,out_text)
    
    out_text = 'Percent Spherical: {:6.2f}'.format(per_sphere) 
    ax4.text(80,0.19,out_text)
    
    out_text = '       N_r (555 nm): {:7.4f}'.format(nr[4]) 
    ax4.text(80,0.16,out_text)
    
    out_text = '       N_i (555 nm): {:7.4f}'.format(ni[4]) 
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