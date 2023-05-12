# -*- coding: utf-8 -*-
"""
Plotting
"""

#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches

#_______________Define ROI Functions___________________________#

def image_crop(a):
        #np.clip(a, 0, None, out=a)
        a[a == -999] = np.nan
        mid_row = a.shape[0] // 2
        mid_col = a.shape[1] // 2
        start_row = mid_row - 262
        end_row = mid_row + 262
        start_col = mid_col - 262
        end_col = mid_col + 262
        
        a = a[start_row:end_row, start_col:end_col]
        return a


def calculate_std(image):
# Define the size of the regions we'll calculate the standard deviation for
    region_size = 5

    # Calculate the standard deviation over the regions
    std_dev = np.zeros_like(image)
    for i in range(region_size//2, image.shape[0] - region_size//2):
        for j in range(region_size//2, image.shape[1] - region_size//2):
            std_dev[i,j] = np.std(image[i-region_size//2:i+region_size//2+1, j-region_size//2:j+region_size//2+1])

    return std_dev


def  choose_roi(image): 
            std_dev = calculate_std(image)
    # Plot the original image and the standard deviation image side by side
            fig, ax = plt.subplots(1,2,  figsize=(16, 8))
            ax[0].imshow(image , cmap = 'gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            im = ax[1].imshow(std_dev, cmap = 'jet')
            ax[1].set_title('Standard Deviation')
            ax[1].grid(True)
            cbar = fig.colorbar(im, ax = ax[1], fraction = 0.046, pad=0.04)
            
            plt.show()

        # Prompt the user to choose a region
            x = int(input('Enter x-coordinate of region: '))
            y = int(input('Enter y-coordinate of region: '))

          
            # Create a new figure with 1 row and 2 columns
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # Plot the original image with the selected region of interest highlighted
            axs[0].imshow(image, cmap='gray')
            axs[0].add_patch(patches.Rectangle((x, y), 5, 5, linewidth=5, edgecolor='w', facecolor='none'))
            axs[0].set_title('Selected Region of Interest')

            # Plot the standard deviation image with the selected region of interest highlighted
            im = axs[1].imshow(std_dev, cmap='jet')
            axs[1].add_patch(patches.Rectangle((x, y),5,5,linewidth=5, edgecolor='w', facecolor='none'))
            axs[1].set_title('Standard Deviation with Selected Region of Interest')
            cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

        # Show the plot
            plt.show()
            
            
            return x,y

#_______________Start the main code____________#
def main():  # Main code

#___________________Section 1: Data Products______________________#

#Load in Data
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/

# Set paths to AirMSPI data and where the output SDATA file should be saved 
# NOTE: datapath is the location of the AirMSPI HDF data files
#       outpath is where the output should be stored
#Work Computer
    datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data2/"
    #datapath = "C:/Users/ULTRASIP_1/Documents/Bakersfield707_DataCopy/"
    outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1123/2FIREX"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Apr1823/Merd_Bakersfield"
    #outpath = "C:/Users/ULTRASIP_1/Desktop/ForGRASP/Retrieval_Files"


# #Home Computer 
#     datapath = "C:/Users/Clarissa/Documents/AirMSPI/Prescott/FIREX-AQ_8172019"
#     outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/Mar1523/1_FIREX"

# Load in the set of measurement sequences
# Set the length of one measurement sequence of step-and-stare observations
# NOTE: This will typically be an odd number (9,7,5,...)

    num_step = 5
    
# Calculate the middle of the sequence

    mid_step = int(num_step/2)  
    
# Set the index of the sequence of step-and-stare files
# NOTE: This is 0 for the first group in the directory, 1 for the second group, etc.

    step_ind = 0
    
# Set the number of wavelengths for radiometric and polarization separately
#num_int = total number of radiometric channels
#num_pol = total number of polarimetric channels

    num_int = 8 
    num_pol = 3
    
# Create arrays to store data
# NOTE: Pay attention to the number of wavelengths
    num_meas = 13

#Measurement Arrays   
    qs_std = np.zeros((num_step,num_pol))  # Qscattering plane
    us_std = np.zeros((num_step,num_pol))  # Uscattering plane
    qm_std = np.zeros((num_step,num_pol))  # Q meridional
    um_std = np.zeros((num_step,num_pol))  # U meridional


    
#____________________Sort the data____________________#

# Change directory to the datapath
    os.chdir(datapath)

# Get the list of files in the directory
# NOTE: Python returns the files in a strange order, so they will need to be sorted by time
    #Search for files with the correct names
    search_str = '*TERRAIN*.hdf'
    dum_list = glob.glob(search_str)
    raw_list = np.array(dum_list)  # Convert to a numpy array
    
# Get the number of files    
    num_files = len(raw_list)
    
# Check the number of files against the index to only read one measurement sequence
    print("AirMSPI Files Found: ",num_files)
    
    num_need = num_step*step_ind+num_step
    
    if(num_need > num_files):
        print("***ERROR***")
        print("Not Enough Files for Group Index")
        print(num_need)
        print(num_files)
        print("Check index of the group at Line 44")
        print("***ERROR***")
        print('error')

# Loop through files within the sequence and sort by time (HHMMSS)
# and extract date and target name
#Filenaming strings 
    #Measurement time as an integer
    time_raw = np.zeros((num_files),dtype=int) 
    
    #Time, Date, and Target Name as a string
    time_str_raw = []  
    date_str_raw = []  
    target_str_raw = [] 

#Start the for loop
    for loop in range(num_files):
    
# Select appropriate file

        this_file = raw_list[loop]

# Parse the filename to get information
    
        words = this_file.split('_')
        
        date_str_raw.append(words[4])
        time_str_raw.append(words[5])  # This will retain the "Z" designation
        target_str_raw.append(words[6])
        
        temp = words[5]
        hold = temp.split('Z')
        time_hhmmss = int(hold[0])
        time_raw[loop] = time_hhmmss

# Convert data to numpy arrays

    date_str = np.array(date_str_raw)
    time_str = np.array(time_str_raw)
    target_str = np.array(target_str_raw)

# Sort the files

    sorted = np.argsort(time_raw)
    mspi_list = raw_list[sorted]
    time_list = time_raw[sorted]
    
    date_str_list = date_str[sorted]
    time_str_list = time_str[sorted]
    target_str_list = target_str[sorted]
    
# Loop through the files for one set of step-and-stare acquisitions

    for loop in range(num_step):
    
        this_ind = loop+num_step*step_ind
        
# Test for the middle of the acquisition sequence

        if(loop == mid_step):
            this_date_str = date_str_list[this_ind]
            this_time_str = time_str_list[this_ind]
            this_target_str = target_str_list[this_ind]
                
# Get the filename

        inputName = datapath+'/'+mspi_list[this_ind]
        
# Tell user location in process

        print("Reading: "+inputName)
        
# Open the HDF-5 file

        f = h5py.File(inputName,'r')
        
#_________________________Read the data_________________________#
# Set the datasets and read (355 nm)
# Radiometric Channel

   
        I_355 = f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]          

# Set the datasets and read (380 nm)
# Radiometric Channel

        I_380 = f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:]      


# Set the datasets and read (445 nm)
# Radiometric Channel


        I_445 = f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:]     

        
# Set the datasets and read (470 nm)
# Polarized band (INCLUDE SOLAR ANGLES)
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane


        I_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:]
        Qs_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/'][:]
        Us_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/'][:]
        Qm_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_meridian/'][:]
        Um_470 = f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_meridian/'][:]

        
# Set the datasets and read (555 nm)
# Radiometric Channel

        I_555 = f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:]     
                
# Set the datasets and read (660 nm)
# Polarized band
# NOTE: GRASP wants polarization in the meridian plane, but this needs to be
#       calculated from the AirMSPI values in the scattering plane

        I_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:]
        Qs_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_scatter/'][:]
        Us_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_scatter/'][:]
        Qm_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_meridian/'][:]
        Um_660 = f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_meridian/'][:]

# Set the datasets and read (865 nm)

        I_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:]
        Qs_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_scatter/'][:]
        Us_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_scatter/'][:]
        Qm_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_meridian/'][:]
        Um_865 = f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_meridian/'][:]
 

    
# Close the file
        f.close()

#_____________________Perform Data Extraction___________________#
# Extract the data in the large bounding box
# NOTE: This puts the array into *image* space       

        img_i_355 = np.flipud(image_crop(I_355))
        img_i_380 = np.flipud(image_crop(I_380))
        img_i_445 = np.flipud(image_crop(I_445))
        img_i_470 = np.flipud(image_crop(I_470))
        img_i_555 = np.flipud(image_crop(I_555))
        img_i_660 = np.flipud(image_crop(I_660))
        img_i_865 = np.flipud(image_crop(I_865))
        
        img_qs_470 = np.flipud(image_crop(Qs_470))
        img_qs_660 = np.flipud(image_crop(Qs_660))
        img_qs_865 = np.flipud(image_crop(Qs_865))
        
        img_us_470 = np.flipud(image_crop(Us_470))
        img_us_660 = np.flipud(image_crop(Us_660))
        img_us_865 = np.flipud(image_crop(Us_865))
        
        img_qm_470 = np.flipud(image_crop(Qm_470))
        img_qm_660 = np.flipud(image_crop(Qm_660))
        img_qm_865 = np.flipud(image_crop(Qm_865))
        
        img_um_470 = np.flipud(image_crop(Um_470))
        img_um_660 = np.flipud(image_crop(Um_660))
        img_um_865 = np.flipud(image_crop(Um_865))
        
        
# Test for valid data    
# NOTE: The test is done for all the wavelengths that are read, so if the wavelengths
#       are changed, then the test needs to change  - note about missing data  
        
        good = ((img_i_355 > 0.0) & (img_i_380 > 0.0) & (img_i_445 > 0.0) &
            (img_i_470 > 0.0) & (img_i_555 > 0.0) & (img_i_660 > 0.0) &
            (img_i_865 > 0.0))
            
        img_good = img_i_355[good]
        
        if(len(img_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print('error')
        
        if loop == 0: 
            x_center, y_center = choose_roi(img_i_470)
        
        box_x1 = x_center - 2 
        box_x2 = x_center + 3 
        box_y1 = y_center - 2 
        box_y2 = y_center + 3 
        
            
# Extract the values from the ROI
# NOTE: The coordinates are relative to the flipped "img" array

        box_i_355 = img_i_355[box_x1:box_x2,box_y1:box_y2]
        box_i_380 = img_i_380[box_x1:box_x2,box_y1:box_y2]
        box_i_445 = img_i_445[box_x1:box_x2,box_y1:box_y2]
        box_i_470 = img_i_470[box_x1:box_x2,box_y1:box_y2]
        box_i_555 = img_i_555[box_x1:box_x2,box_y1:box_y2]
        box_i_660 = img_i_660[box_x1:box_x2,box_y1:box_y2]
        box_i_865 = img_i_865[box_x1:box_x2,box_y1:box_y2]
             
        box_qs_470 = img_qs_470[box_x1:box_x2,box_y1:box_y2]
        box_qs_660 = img_qs_660[box_x1:box_x2,box_y1:box_y2]
        box_qs_865 = img_qs_865[box_x1:box_x2,box_y1:box_y2]
        
        
        box_us_470 = img_us_470[box_x1:box_x2,box_y1:box_y2]
        box_us_660 = img_us_660[box_x1:box_x2,box_y1:box_y2]
        box_us_865 = img_us_865[box_x1:box_x2,box_y1:box_y2]
        

        box_qm_470 = img_qm_470[box_x1:box_x2,box_y1:box_y2]
        box_qm_660 = img_qm_660[box_x1:box_x2,box_y1:box_y2]
        box_qm_865 = img_qm_865[box_x1:box_x2,box_y1:box_y2]
        
        
        box_um_470 = img_um_470[box_x1:box_x2,box_y1:box_y2]
        box_um_660 = img_um_660[box_x1:box_x2,box_y1:box_y2]
        box_um_865 = img_um_865[box_x1:box_x2,box_y1:box_y2]


        

        
# Extract the valid data and calculate the std
# NOTE: The test is done for all the wavelengths that are read, so if the wavelengths
#       are changed, then the test needs to change    
        
        good = ((box_i_355 > 0.0) & (box_i_380 > 0.0) & (box_i_445 > 0.0) &
            (box_i_470 > 0.0) & (box_i_555 > 0.0) & (box_i_660 > 0.0) &
            (box_i_865 > 0.0))
            
        box_good = box_i_355[good]

        
        if(len(box_good) < 1):
            print("***ERROR***")
            print("NO VALID PIXELS")
            print("***ERROR***")
            print('error')

        
        qs_470 = np.std(box_qs_470[good])
        qs_660 = np.std(box_qs_660[good])
        qs_865 = np.std(box_qs_865[good])
        
        us_470 = np.std(box_us_470[good])
        us_660 = np.std(box_us_660[good])
        us_865 = np.std(box_us_865[good])
        
        
        qm_470 = np.std(box_qm_470[good])
        qm_660 = np.std(box_qm_660[good])
        qm_865 = np.std(box_qm_865[good])
        
        
        um_470 = np.std(box_um_470[good])
        um_660 = np.std(box_um_660[good])
        um_865 = np.std(box_um_865[good])
        
            
        qs_std[loop,0] = qs_470
        qs_std[loop,1] = qs_660
        qs_std[loop,2] = qs_865
        
        us_std[loop,0] = us_470
        us_std[loop,1] = us_660
        us_std[loop,2] = us_865
        
        qm_std[loop,0] = qm_470
        qm_std[loop,1] = qm_660
        qm_std[loop,2] = qm_865
        
        um_std[loop,0] = um_470
        um_std[loop,1] = um_660
        um_std[loop,2] = um_865
        
    return qs_std,us_std,qm_std,um_std

### END MAIN FUNCTION
if __name__ == '__main__':
    qs,us,qm,um = main() 
     
#def plotout(qs,us,qm,um):  # Main code

# Set the paths
# NOTE: basepath is the location of the GRASP output files
#       figpath is where the image output should be stored


    basepath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1123/2FIREX"
    figpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1123/2FIREX/Plots"

    # basepath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/1FIREXR2"
    # figpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/Retrievals/May1023/1FIREXR2/Plots"

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
    
    
    fig, ax = plt.subplots(3, 3,
                           sharex=True, 
                           sharey=False,figsize=(18, 10), dpi=560)

    ax[0,0].plot(scat[:,0],i_obs[:,0],linestyle='dashed',color="green",linewidth='3')
    ax[0,0].plot(scat[:,1],i_obs[:,1],linestyle='dashed',color="pink",linewidth='3')
    ax[0,0].plot(scat[:,2],i_obs[:,2],linestyle='dashed',color="brown",linewidth='3')  
    ax[0,0].plot(scat[:,4],i_obs[:,4],linestyle='dashed',color="purple",linewidth='3')
    
    ax[0,0].set_ylabel('BRF(I)', fontsize='x-large')
    ax[0,0].yaxis.set_label_coords(-.2, .5)

    
    ax[0,0].plot(scat[:,0],i_mod[:,0],color="green",label="355",linewidth='3')
    ax[0,0].plot(scat[:,1],i_mod[:,1],color="pink",label="380",linewidth='3')
    ax[0,0].plot(scat[:,2],i_mod[:,2],color="brown",label="445",linewidth='3')
    ax[0,0].plot(scat[:,4],i_mod[:,4],color="purple",label="555",linewidth='3')
    
    ax[0,0].set_yticks([0.04,0.06,0.1,0.15,0.2])
    ax[0,0].yaxis.set_tick_params(labelsize=15)
    
    ax[0,1].plot(scat[:,3],i_obs[:,3],linestyle='dashed',color="blue",linewidth='3')
    ax[0,1].plot(scat[:,5],i_obs[:,5],linestyle='dashed',color="red",linewidth='3')
    ax[0,1].plot(scat[:,6],i_obs[:,6],linestyle='dashed',color="orange",linewidth='3')
    ax[0,1].plot(scat[:,3],i_mod[:,3],color="blue",label='470',linewidth='3')
    ax[0,1].plot(scat[:,5],i_mod[:,5],color="red",label='660',linewidth='3')
    ax[0,1].plot(scat[:,6],i_mod[:,6],color="orange",label='865',linewidth='3')
    
    ax[0,1].set_yticks([0.02,0.04,0.06,0.08,0.1])
    ax[0,1].yaxis.set_tick_params(labelsize=15)
    
    ax[0,0].legend(bbox_to_anchor=(3.25,0.80),
          ncol=2,fontsize='xx-large',frameon=False)
    ax[0,1].legend(bbox_to_anchor=(1.28,0.5),
          ncol=2,fontsize='xx-large',frameon=False)
    
    
    ax[1,0].set_ylabel('BRF(Q)', fontsize='x-large')
    ax[1,0].yaxis.set_label_coords(-.2, .5)
    
    
    ax[1,0].plot(scat[:,3],q_obs[:,0],linestyle='dashed',color="blue",label="470nm",linewidth='3')
    ax[1,0].errorbar(scat[:,3],q_obs[:,0],qs[:,0], marker = 'd',alpha=0.4,color='blue',markersize=15)
    
    ax[1,0].set_yticks([0.30,0.35,0.40,0.45,0.50])
    ax[1,0].yaxis.set_tick_params(labelsize=15)

    ax[1,1].plot(scat[:,5],q_obs[:,1],linestyle='dashed',color="red",label="660nm",linewidth='3')
    ax[1,1].errorbar(scat[:,5],q_obs[:,1],qs[:,1],marker = 'd',alpha=0.4,color='red',markersize=15)
    
    ax[1,1].set_yticks([0.15,0.25,0.30,0.35,0.40])
    ax[1,1].yaxis.set_tick_params(labelsize=15)

    ax[1,2].plot(scat[:,6],q_obs[:,2],linestyle='dashed',color="orange",label="865nm",linewidth='3')
    ax[1,2].errorbar(scat[:,6],q_obs[:,2],qs[:,2], marker = 'd',alpha=0.6,color='orange',markersize=15)
    
    ax[1,2].set_yticks([0.05,0.1,0.13,0.15,0.35])
    ax[1,2].yaxis.set_tick_params(labelsize=15)

    
    ax[1,0].plot(scat[:,3],q_mod[:,0],color="blue",linewidth='3')
    ax[1,1].plot(scat[:,5],q_mod[:,1],color="red",linewidth='3')
    ax[1,2].plot(scat[:,6],q_mod[:,2],color="orange",linewidth='3')
    
    fig.delaxes(ax[0,2])
    
    ax[2,0].set_ylabel('BRF(U)', fontsize='x-large')
    ax[2,0].yaxis.set_label_coords(-0.2, .5)
    
    ax[2,0].plot(scat[:,3],u_obs[:,0],linestyle='dashed',color="blue",label="470nm",linewidth='3')
    ax[2,0].errorbar(scat[:,3],u_obs[:,0],us[:,0], marker = 'd',alpha=0.4,color='blue',markersize=15)
    
    ax[2,0].set_yticks([-0.06,-0.04,-0.02,0,0.1])
    ax[2,0].yaxis.set_tick_params(labelsize=15)

    ax[2,1].plot(scat[:,5],u_obs[:,1],linestyle='dashed',color="red",label="660nm",linewidth='3')
    ax[2,1].errorbar(scat[:,5],u_obs[:,1],us[:,1],marker = 'd',alpha=0.4,color='red',markersize=15)
    
    ax[2,1].set_yticks([-0.04,-0.03,-0.02,-0.01,0])
    ax[2,1].yaxis.set_tick_params(labelsize=15)

    ax[2,2].plot(scat[:,6],u_obs[:,2],linestyle='dashed',color="orange",label="865nm",linewidth='3')
    ax[2,2].errorbar(scat[:,6],u_obs[:,2],us[:,2], marker = 'd',alpha=0.6,color='orange',markersize=15)
    
    ax[2,2].set_yticks([-0.01,-0.008,-0.004,0,0.003])
    ax[2,2].yaxis.set_tick_params(labelsize=15)


    ax[2,0].plot(scat[:,3],u_mod[:,0],color="blue",linewidth='3')
    ax[2,1].plot(scat[:,5],u_mod[:,1],color="red",linewidth='3')
    ax[2,2].plot(scat[:,6],u_mod[:,2],color="orange",linewidth='3')
    

    ax[2,1].set_xlabel(u"\u03A9 [\u00b0]",fontsize='xx-large')   
    ax[2,1].xaxis.set_tick_params(labelsize=15)
    ax[2,2].xaxis.set_tick_params(labelsize=15)
    ax[2,0].xaxis.set_tick_params(labelsize=15)
    
    plt.show()
    
# ### END MAIN FUNCTION
# if __name__ == '__plotout__':
#      plotout(qs,us,qm,um) 
