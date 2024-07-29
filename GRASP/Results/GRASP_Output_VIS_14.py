# GRASP_Output_VIS_14.py
#
# This is a Python 3.6.9 code to read the output of the GRASP retrieval code
# and plot the results.
#
# Creation Date: 2019-11-08
# Last Modified: 2019-11-08
#
# by Michael J. Garay
# (Michael.J.Garay@jpl.nasa.gov)

# Import packages

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def main():  # Main code

# Time the code

    start_time = time.time()

# Set the paths

    #basepath = '/Users/mgaray/Desktop/Radiative_Transfer/RT_CODES/grasp-master/APOLO/2019_11_08/'
    basepath = 'C:/Users/ULTRASIP_1/OneDrive/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Aug1923'
    #figpath = '/Users/mgaray/Desktop/CODING/PYTHON/PY36/NOV19/APOLO/FIGS/'
    figpath = 'C:/Users/ULTRASIP_1/OneDrive/Documents/ULTRASIP/AirMSPI_FIREXAQ/Figures'
# Set the base output file name

    #out_base = '_AirMSPI_test_20b_14.png'
    out_base = '_AirMSPI_test.png'
# Change directory to the basepath

    os.chdir(basepath)

# Get the text file listing

    file_list = glob.glob('*Merd_R5*.txt')
    #file_list = glob.glob('bench_FWD_IQU_rslts.txt')
    
    num_files = len(file_list)
    
    print("FOUND FILES:",num_files)
    
### READ DATA

# Tell user location in process

    inputName = file_list[0]
    print("Reading: "+inputName)

# Set arrays to store the data

    num_size_bins = 22
    size_bin = np.zeros(num_size_bins)
    size_bin_val = np.zeros(num_size_bins)
    size_bin_count = 0
    size_bin_0 = 9999
    
    cv_fit = np.zeros(3)
    rv_fit = np.zeros(3)
    std_fit = np.zeros(3)
    cv_fit_count = 0
    rv_fit_count = 0
    std_fit_count = 0
    
    per_bin_0 = 9999
    ae_0 = 9999
    
    num_wave = 2
    wave = np.zeros(num_wave)
    aod = np.zeros(num_wave)
    ssa = np.zeros(num_wave)
    nr = np.zeros(num_wave)
    ni = np.zeros(num_wave)
    wave_count = 0
    aod_0 = 9999
    ssa_0 = 9999
    nr_0 = 9999
    ni_0 = 9999

# NOTE: Do not know how long these fields will be
    
    band1_scat_raw = []
    band1_meas_I_raw = []
    band1_fit_I_raw = []
    band1_meas_q_raw = []
    band1_fit_q_raw = []
    band1_meas_u_raw = []
    band1_fit_u_raw = []
    
    band2_scat_raw = []
    band2_meas_I_raw = []
    band2_fit_I_raw = []
    band2_meas_q_raw = []
    band2_fit_q_raw = []
    band2_meas_u_raw = []
    band2_fit_u_raw = []
    
    fit_counter_1 = 0
    fit_counter_2 = 0
    band1_0 = 10000
    band2_0 = 20000
    band3_0 = 30000

# Set a counter

    data_count = 0

# Read the file
# NOTE: Deprecated using readlines() based on the information here:
# http://stupidpythonideas.blogspot.com/2013/06/readlines-considered-silly.html
# Additionally, changed the file opening and handling

    with open(inputName,'r') as inputFile:

# Read the lines

        for line in inputFile:
        
# Check for lines with more than zero elements        
        
            if(len(line) > 1):

# Parse the line on blanks

                words = line.split()
            
# Look for size distribution

                if(words[0] == 'Size'):
                    print("FOUND SIZE BINS")
                    size_bin_0 = data_count

# Look for the size parameters

                if(words[0] == 'cv'):
                    cv_fit[cv_fit_count] = float(words[2])
                    cv_fit_count = cv_fit_count + 1
                    if(cv_fit_count > 3):
                       print('error')
                       
                if(words[0] == 'rv'):
                    rv_fit[rv_fit_count] = float(words[2])
                    rv_fit_count = rv_fit_count + 1
                    if(rv_fit_count > 3):
                       print('error')
                       
                if(words[0] == 'std'):
                    std_fit[std_fit_count] = float(words[2])
                    std_fit_count = std_fit_count + 1
                    if(std_fit_count > 3):
                       print('error')
                       
# Look for percentage of spherical particles

                if(words[0] == '%'):
                    per_bin_0 = data_count  
                    
# Look for Angstrom Exponent

                if(words[0] == 'Angstrom'):
                    ae_0 = data_count  
                    
# Look for spectral information

                if(words[0] == 'Wavelength'):
                    if(words[2] == 'AOD_Total'):
                        aod_0 = data_count
                    if(words[2] == 'SSA_Total'):
                        ssa_0 = data_count 
                    if(words[2] == 'REAL'):
                        nr_0 = data_count
                    if(words[2] == 'IMAG'):
                        ni_0 = data_count
                        
# Look for the fitting information
# NOTE: Do not know how long these fields will need to be, so set up as dummy fields

                if(words[0] == '#'):
                    if(band1_0 == 10000):
                        band1_0 = data_count
                    if(band2_0 == 10000):
                        band2_0 = data_count
                    if(band3_0 == 10000):
                        band3_0 = data_count

# Extract the size distribution information

            if(data_count-size_bin_0 > 0):
                size_bin[size_bin_count] = float(words[0])
                size_bin_val[size_bin_count] = float(words[1])
                size_bin_count = size_bin_count+1
                if(size_bin_count == num_size_bins):
                    size_bin_0 = 9999
                    
# Extract the spectral AOD information

            if(data_count-aod_0 > 0):
                wave[wave_count] = float(words[0])
                aod[wave_count] = float(words[1])
                wave_count = wave_count+1
                if(wave_count == num_wave):
                    aod_0 = 9999
                    wave_count = 0
                    
# Extract the spectral SSA information

            if(data_count-ssa_0 > 0):
                ssa[wave_count] = float(words[1])
                wave_count = wave_count+1
                if(wave_count == num_wave):
                    ssa_0 = 9999
                    wave_count = 0
                    
# Extract the spectral n_r information
# NOTE: Have to skip a line

            if(data_count-nr_0 > 1):
                nr[wave_count] = float(words[1])
                wave_count = wave_count+1
                if(wave_count == num_wave):
                    nr_0 = 9999
                    wave_count = 0
                    
# Extract the spectral n_i information
# NOTE: Have to skip a line

            if(data_count-ni_0 > 1):
                ni[wave_count] = float(words[1])
                wave_count = wave_count+1
                if(wave_count == num_wave):
                    ni_0 = 9999
                    wave_count = 0
                    
# Extract the percentage of spherical particles

            if(data_count-per_bin_0 > 0):
                percent_sphere = float(words[1])
                per_bin_0 = 9999
                
# Extract the Angstrom Exponent

            if(data_count-ae_0 > 0):
                ae = float(words[0])
                ae_0 = 9999

# Extract the fitting information

            if(data_count-band1_0 > 0):
                if(line[0:10] == '----------'):
                    band1_0 = 9999
                    band2_0 = 10000
                elif(line[0:4] == '   #'):
                    fit_counter_1 = fit_counter_1+1
                    continue
                else:
                    if(fit_counter_1 == 0):
                        scat_temp = float(words[4])
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band1_scat_raw.append(scat_temp)
                        band1_meas_I_raw.append(meas_temp)
                        band1_fit_I_raw.append(fit_temp)
                    if(fit_counter_1 == 1):
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band1_meas_q_raw.append(meas_temp)
                        band1_fit_q_raw.append(fit_temp)
                    if(fit_counter_1 == 2):
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band1_meas_u_raw.append(meas_temp)
                        band1_fit_u_raw.append(fit_temp)
                    
            if(data_count-band2_0 > 0):
                if(line[0:10] == '----------'):
                    band2_0 = 9999
                    band3_0 = 10000
                elif(line[0:4] == '   #'):
                    fit_counter_2 = fit_counter_2+1
                    continue
                elif(len(line) == 1):
                    continue
                elif(words[0] == 'INVSING'):
                    continue
                else:
                    if(fit_counter_2 == 0):
                        scat_temp = float(words[4])
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band2_scat_raw.append(scat_temp)
                        band2_meas_I_raw.append(meas_temp)
                        band2_fit_I_raw.append(fit_temp)
                    if(fit_counter_2 == 1):
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band2_meas_q_raw.append(meas_temp)
                        band2_fit_q_raw.append(fit_temp)
                    if(fit_counter_2 == 2):
                        meas_temp = float(words[5])
                        fit_temp = float(words[6])
                        band2_meas_u_raw.append(meas_temp)
                        band2_fit_u_raw.append(fit_temp)
                    
            if(data_count-band3_0 > 0):
                if(len(line) == 1):  #  This set does not have the same separator
                    band3_0 = 9999
                else:
                    scat_temp = float(words[4])
                    meas_temp = float(words[5])
                    fit_temp = float(words[6])
                    # band3_scat_raw.append(scat_temp)
                    # band3_meas_raw.append(meas_temp)
                    # band3_fit_raw.append(fit_temp)

# Increment the counter
            
            data_count = data_count+1

### PLOT THE DATA

# Change to the figure directory

    os.chdir(figpath)

### FIRST PLOT (Size Distribution)

# Set the plot area (using the concise format)

    fig, ax = plt.subplots(figsize=(9,6), dpi=120)

# Calculate the fits

    r = np.arange(1000000)+1
    r = r/10000.

# Calculate the scaling factors

    sff = cv_fit[1]/(cv_fit[1]+cv_fit[2])
    sfc = cv_fit[2]/(cv_fit[1]+cv_fit[2])

# Fine mode    

    brack = (np.log(r)-np.log(rv_fit[1]))**2/(2.*(std_fit[1])**2)
#    coeff = cv_fit[1]/(np.sqrt(2.*np.pi)*(std_fit[1]))
#    fmv = coeff*np.exp(-1.*brack)*sff*slop
    coeff = 1.0/(np.sqrt(2.*np.pi)*(std_fit[1]))
    fmv = coeff*np.exp(-1.*brack)*sff
    print("FITTING")
    print(cv_fit[1])
    print(cv_fit[2])
    
# Coarse mode    

    brack = (np.log(r)-np.log(rv_fit[2]))**2/(2.*(std_fit[2])**2)
#    coeff = cv_fit[2]/(np.sqrt(2.*np.pi)*(std_fit[2]))
#    cmv = coeff*np.exp(-1.*brack)*sfc*slop
    coeff = 1.0/(np.sqrt(2.*np.pi)*(std_fit[2]))
    cmv = coeff*np.exp(-1.*brack)*sfc

# Plot the data

    print(np.amin(size_bin_val))
    print(np.amax(size_bin_val))
        
    plt.plot(size_bin,size_bin_val,color="green",marker='o',linestyle='dashed',linewidth=2, markersize=10)
    plt.plot(r,fmv,color="blue")
    plt.plot(r,cmv,color="red")

    plt.xlim(0.01,15)
    plt.xticks(np.arange(0.01,15,1))
    plt.xlabel('Radius (Microns)')
    plt.xscale('log')
    
    plt.ylim(0.0,1.1)
    plt.yticks(np.arange(0.0,1.2,0.1))
    plt.ylabel('Volume Weighting dV(r)/dln(r)')
    
    plt.title('GRASP Retrieved Size Bins and Size Distributions')

# Save the file    
    
    outfile = 'Size_Dist'+out_base
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 

### SECOND PLOT (Spectral Information)

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(9,6),dpi=120)

# First subplot (AOD vs. wavelength)
    
    aod_plot_max = 0.5
    ax1.plot(wave,aod,color="k",marker='o',linestyle='dashed',linewidth=2, markersize=7)
    ax1.set_ylim(0.0,aod_plot_max)
    
    ax1.set_title("Retrieved AOD vs. Wavelength")
    ax1.set(xlabel="Wavelength (Microns)",ylabel="AOD")
    out_text = f'AE = {ae:.2f}'
    ax1.text(0.90*wave[num_wave-1],0.8*aod_plot_max,out_text)
    out_text = f'Sphere = {percent_sphere:.1f}%'
    ax1.text(0.90*wave[num_wave-1],0.73*aod_plot_max,out_text)
    
# Second subplot (SSA vs. wavelength)

    ssa_plot_min = 0.97
    ax2.plot(wave,ssa,color="k",marker='o',linestyle='dashed',linewidth=2, markersize=7)
    ax2.set_ylim(ssa_plot_min,1.0)
    ax2.set_title("Retrieved SSA vs. Wavelength")
    ax2.set(xlabel="Wavelength (Microns)",ylabel="SSA")
    
# Third subplot (n_r vs. wavelength)

    nr_plot_min = 1.3
    nr_plot_max = 1.6
    ax3.plot(wave,nr,color="k",marker='o',linestyle='dashed',linewidth=2, markersize=7)
    ax3.set_ylim(nr_plot_min,nr_plot_max)
    ax3.set_title("Real Refractive Index vs. Wavelength")
    ax3.set(xlabel="Wavelength (Microns)",ylabel="n_r")
    
# Fourth subplot (n_i vs. wavelength)

    ni_plot_min = 0.0010
    ni_plot_max = 0.0030
    ax4.plot(wave,ni,color="k",marker='o',linestyle='dashed',linewidth=2, markersize=7)
    ax4.set_ylim(ni_plot_min,ni_plot_max)
    ax4.set_title("Imaginary Refractive Index vs. Wavelength")
    ax4.set(xlabel="Wavelength (Microns)",ylabel="n_i")

# Set tight layout

    plt.tight_layout()
    
# Save the file    
    
    outfile = 'Spectral'+out_base
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300)   
    
#    plt.close()

### THIRD PLOT (Fitting)

# Set the plot area (using the concise format)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(9,6),dpi=120)

# Convert data to numpy arrays

    band1_scat = np.array(band1_scat_raw)
    band1_meas_I = np.array(band1_meas_I_raw)
    band1_fit_I = np.array(band1_fit_I_raw)
    band1_meas_q = np.array(band1_meas_q_raw)
    band1_fit_q = np.array(band1_fit_q_raw)
    band1_meas_u = np.array(band1_meas_u_raw)
    band1_fit_u = np.array(band1_fit_u_raw)
    
    band2_scat = np.array(band2_scat_raw)
    band2_meas_I = np.array(band2_meas_I_raw)
    band2_fit_I = np.array(band2_fit_I_raw)
    band2_meas_q = np.array(band2_meas_q_raw)
    band2_fit_q = np.array(band2_fit_q_raw)
    band2_meas_u = np.array(band2_meas_u_raw)
    band2_fit_u = np.array(band2_fit_u_raw)
    
# Calculate DOLP

    band1_meas_dolp = np.sqrt(band1_meas_q**2 + band1_meas_u**2)
    band1_fit_dolp = np.sqrt(band1_fit_q**2 + band1_fit_u**2)
    band2_meas_dolp = np.sqrt(band2_meas_q**2 + band2_meas_u**2)
    band2_fit_dolp = np.sqrt(band2_fit_q**2 + band2_fit_u**2)
    
# First subplot (I)

    ax1.plot(band1_scat,band1_meas_I,color='red',linestyle='solid',linewidth=1,label='Meas')
    ax1.plot(band1_scat,band1_fit_I,color='red',linestyle='dashed',linewidth=1,label='Fit')
    ax1.plot(band2_scat,band2_meas_I,color='magenta',linestyle='solid',linewidth=1,label='Meas')
    ax1.plot(band2_scat,band2_fit_I,color='magenta',linestyle='dashed',linewidth=1,label='Fit')

    ax1.set_xlim(90.0,160)
    ax1.set_xticks(np.arange(90,170,10))
      
    ax1.set_ylim(0.0,0.10)
    ax1.set_yticks(np.arange(0.0,0.11,0.01))
    ax1.set(xlabel='Scattering Angle (Deg)',ylabel='Intensity')

    ax1.legend(loc='upper right')
    
    ax1.set_title('GRASP Fitting Results I')
    
# Second subplot (q)

    ax2.plot(band1_scat,band1_meas_q,color='red',linestyle='solid',linewidth=1,label='Meas')
    ax2.plot(band1_scat,band1_fit_q,color='red',linestyle='dashed',linewidth=1,label='Fit')
    ax2.plot(band2_scat,band2_meas_q,color='magenta',linestyle='solid',linewidth=1,label='Meas')
    ax2.plot(band2_scat,band2_fit_q,color='magenta',linestyle='dashed',linewidth=1,label='Fit')

    ax2.axhline(y=0,color='k')

    ax2.set_xlim(90.0,160)
    ax2.set_xticks(np.arange(90,170,10))
      
    ax2.set_ylim(-0.6,0.6)
#    ax2.set_yticks(np.arange(0.0,0.16,0.05))
    ax2.set(xlabel='Scattering Angle (Deg)',ylabel='Q/I')
    
    ax2.set_title('GRASP Fitting Results Q/I')
    
# Third subplot (DOLP)

    ax3.plot(band1_scat,band1_meas_dolp,color='red',linestyle='solid',linewidth=1,label='Meas')
    ax3.plot(band1_scat,band1_fit_dolp,color='red',linestyle='dashed',linewidth=1,label='Fit')
    ax3.plot(band2_scat,band2_meas_dolp,color='magenta',linestyle='solid',linewidth=1,label='Meas')
    ax3.plot(band2_scat,band2_fit_dolp,color='magenta',linestyle='dashed',linewidth=1,label='Fit')

    ax3.set_xlim(90.0,160)
    ax3.set_xticks(np.arange(90,170,10))
      
    ax3.set_ylim(0.0,0.6)
    ax3.set_yticks(np.arange(0.0,0.7,0.1))
    ax3.set(xlabel='Scattering Angle (Deg)',ylabel='DoLP')
    
    ax3.set_title('GRASP Fitting Results DoLP')
    
# Fourth subplot (u)

    ax4.plot(band1_scat,band1_meas_u,color='red',linestyle='solid',linewidth=1,label='Meas')
    ax4.plot(band1_scat,band1_fit_u,color='red',linestyle='dashed',linewidth=1,label='Fit')
    ax4.plot(band2_scat,band2_meas_u,color='magenta',linestyle='solid',linewidth=1,label='Meas')
    ax4.plot(band2_scat,band2_fit_u,color='magenta',linestyle='dashed',linewidth=1,label='Fit')

    ax4.axhline(y=0,color='k')

    ax4.set_xlim(90.0,160)
    ax4.set_xticks(np.arange(90,170,10))
      
    ax4.set_ylim(-0.6,0.6)
#    ax2.set_yticks(np.arange(0.0,0.16,0.05))
    ax4.set(xlabel='Scattering Angle (Deg)',ylabel='U/I')
    
    ax4.set_title('GRASP Fitting Results U/I')

# Set tight layout

    plt.tight_layout()
    
# Save the file    
    
    outfile = 'Fit'+out_base
    print("Saving: "+outfile)
    plt.savefig(outfile,dpi=300) 
    
    plt.show()
 
# Tell user completion was successful

    print("\nSuccessful Completion\n")

### END MAIN FUNCTION


if __name__ == '__main__':
    main()    
