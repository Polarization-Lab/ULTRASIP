# -*- coding: utf-8 -*-

#_______________Import Packages_________________#
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def format_directory():
    
    directory_path = input("Enter the directory path: ")
    # Remove trailing slashes
    directory_path = directory_path.rstrip(os.path.sep)

    # Normalize path separators
    directory_path = os.path.normpath(directory_path)

    # Replace backslashes with forward slashes
    directory_path = directory_path.replace('\\', '/')

    # Format the directory path as a raw string literal
    formatted_directory_path = r"{}".format(directory_path)

    return formatted_directory_path


def load_hdf_files(folder_path, group_size,idx):
    # Get a list of HDF file paths in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*.hdf'))

    # Sort file paths based on date and time in the file name
    sorted_file_paths = sorted(file_paths, key=lambda x: (os.path.basename(x).split('_')[5][0:8], os.path.basename(x).split('_')[5][8:14]))

    # Load files in groups of the specified size
    groups = [sorted_file_paths[i:i+group_size] for i in range(0, len(sorted_file_paths), group_size)]

    # Choose one group to use
    selected_group = groups[idx]  # Change the index to select a different group

    return selected_group

def image_format(image):
    
    image[np.where(image == -999)] = np.nan
    new_image = image[1100:2800,500:2200]
    
    return new_image

#Variable Defns
num_step = 5
step_ind = 0
esd = 0.0
num_int = 8 
num_pol = 3
num_meas = 13

# Create arrays to store data
# ALL ANGLES IN RADIANS
scat_median = np.zeros((num_step,num_int))  # Scattering angle
vza_median = np.zeros((num_step,num_meas))  # View zenith angle
raz_median = np.zeros((num_step,num_meas))  # Relative azimuth angle
sza_median = np.zeros(num_step)  # Solar zenith angle (one per stare)

#Measurement Arrays   
i_median = np.zeros((num_step,num_int))  # Intensity
i_in_polar_median = np.zeros((num_step,num_pol))  # I in polarized bands
qd_median = np.zeros((num_step,num_pol))  # Qscattering plane
ud_median = np.zeros((num_step,num_pol))  # Uscattering plane
q_median = np.zeros((num_step,num_pol))  # Q meridional
u_median = np.zeros((num_step,num_pol))  # U meridional
ipol_median = np.zeros((num_step,num_pol))  # Ipol
dolp_median = np.zeros((num_step,num_pol))  # DoLP

#Center point Arrays
center_wave = np.zeros(num_int)  # Center wavelengths  
center_pol = np.zeros(num_pol)  # Center wavelengths (polarized only)


datapath = format_directory()
outpath = format_directory()
pol_ref_plane = input("Enter polarimetric ref plane (Scattering or Meridian): ")

data_files = load_hdf_files(datapath, num_step,step_ind)

nadir = [string for string in data_files if '000N' in string][0]

f = h5py.File(nadir,'r')

i_470 = image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:])
i_555 = image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:])
i_660 = image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:])

i_rgb = i_470+i_555+i_660

plt.figure(figsize=(10, 10))
plt.imshow(i_rgb, cmap = 'jet')
plt.title('I_RGB at Nadir')
plt.grid(True)
plt.colorbar()

# Set the number of gridlines for both the x-axis and y-axis
num_gridlines_x = 60  # Adjust this number as needed
num_gridlines_y = 60  # Adjust this number as needed

# Adjust the number of gridlines using locator_params
plt.gca().xaxis.get_major_locator().set_params(nbins=num_gridlines_x)
plt.gca().yaxis.get_major_locator().set_params(nbins=num_gridlines_y)

plt.gca().tick_params(axis='x', labelrotation=45)

plt.show()

# Get user input for the (x, y) coordinates
try:
    
    
    row = int(input("Enter the row-coordinate: "))
    column= int(input("Enter the column-coordinate: "))
except ValueError:
    print("Invalid input. Please enter integer values for coordinates.")
    exit()

# Now, plot the image with the black square marker on top
plt.figure(figsize=(10, 10))
plt.imshow(i_rgb, cmap='jet')
plt.title('I_RGB at Nadir')
plt.grid(True)
plt.colorbar()

# Plot a marker at the chosen (row, column) coordinate on top of the I_355 image
plt.scatter(column, row, c='black', marker='s', facecolors='none', edgecolors='black', s=100)

plt.show()

#Define ROI 
del_ROI = 5
row_upper = row+del_ROI
row_lower = row-del_ROI
column_upper = column+del_ROI
column_lower = column-del_ROI
idx=-1

loop = len(data_files)
for i in range(loop):
    
    f = h5py.File(data_files[i],'r')
    
    i_355= np.nanmean(image_format(f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
    plt.figure()
    plt.imshow(image_format(f['/HDFEOS/GRIDS/355nm_band/Data Fields/I/'][:]),cmap='jet')

    if np.isnan(i_355):
        print("NaN values")
        num_step = num_step-1
        continue
    else:     
        #get metadata
        idx = idx + 1
        words = nadir.split('_')
        date = words[5][0:4] +'-'+words[5][4:6]+'-'+words[5][6:8]
        time = words[6][0:2]+':'+words[6][2:4]+':'+words[6][4:8]
        elev = image_format(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Elevation/'][:])[row,column]
        lat = image_format(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude/'][:])[row,column]
        lon = image_format(f['/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude/'][:])[row,column]
        
# Get the Earth-Sun distance from the file attributes from the first file
        if(esd == 0.0):
            print("GETTING EARTH-SUN DISTANCE")
            esd = f['/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/'].attrs['Sun distance']
           
# Get the actual center wavelengths and E0 values
            center_raw = f['/Channel_Information/Center_wavelength/'][:]   
            E0_wave = f['/Channel_Information/Solar_irradiance_at_1_AU/'][:]
            
# Calculate the effective center wavelengths by appropriate averaging
# NOTE: Essentially, for the radiometric only bands, the center wavelength is given in the
#       file. For polarized bands, we average the three available bands.

            center_wave[0] = center_raw[0]  # 355 nm
            center_wave[1] = center_raw[1]  # 380 nm
            center_wave[2] = center_raw[2]  # 445 nm          
            center_wave[3] = (center_raw[3]+center_raw[4]+center_raw[5])/3.0 # 470 nm
            center_wave[4] = center_raw[6]  # 555 nm       
            center_wave[5] = (center_raw[7]+center_raw[8]+center_raw[9])/3.0 # 660 nm
            center_wave[6] = (center_raw[10]+center_raw[11]+center_raw[12])/3.0 # 865 nm
            center_wave[7] = center_raw[13]  # 935 nm          
            
            center_pol[0] = center_wave[3]
            center_pol[1] = center_wave[5]
            center_pol[2] = center_wave[6]
            
# Calculate the effective E0 values by appropriate averaging
# NOTE: Essentially, for radiomentric only bands, the E0 is given in the
#       file. For polarized bands, we average the E0's from the three available bands.

            E0_355 = E0_wave[0]  # 355 nm
            E0_380 = E0_wave[1]  # 380 nm
            E0_445 = E0_wave[2]  # 440 nm
            E0_470 = (E0_wave[3]+E0_wave[4]+E0_wave[5])/3.0 # 470 nm        
            E0_555 = E0_wave[6]  # 555 nm        
            E0_660 = (E0_wave[7]+E0_wave[8]+E0_wave[9])/3.0 # 660 nm
            E0_865 = (E0_wave[10]+E0_wave[11]+E0_wave[12])/3.0 # 865 nm       
            E0_935 = E0_wave[13]  # 935 nm
    
        vaz_355 = np.nanmean(image_format(f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])  
        vza_355 = np.nanmean(image_format(f['/HDFEOS/GRIDS/355nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        
        i_380= np.nanmean(image_format(f['/HDFEOS/GRIDS/380nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_380 = np.nanmean(image_format(f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper]) 
        vza_380 = np.nanmean(image_format(f['/HDFEOS/GRIDS/380nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])   
        
        i_445= np.nanmean(image_format(f['/HDFEOS/GRIDS/445nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_445 = np.nanmean(image_format(f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        vza_445 = np.nanmean(image_format(f['/HDFEOS/GRIDS/445nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        
        i_470= np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        saz = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/Sun_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        sza = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/Sun_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        qs_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        us_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        qm_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/Q_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        um_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/U_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        vza_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        
        scat_355 = np.nanmean(image_format(f['/HDFEOS/GRIDS/355nm_band/Data Fields/Scattering_angle/'][:])[row_lower:row_upper,column_lower:column_upper])
        scat_470 = np.nanmean(image_format(f['/HDFEOS/GRIDS/470nm_band/Data Fields/Scattering_angle/'][:])[row_lower:row_upper,column_lower:column_upper])
        scat_555 = np.nanmean(image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/Scattering_angle/'][:])[row_lower:row_upper,column_lower:column_upper])
        scat_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/Scattering_angle/'][:])[row_lower:row_upper,column_lower:column_upper])

        print(scat_355,scat_470,scat_555,scat_660)
        
        i_555= np.nanmean(image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_555 = np.nanmean(image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        vza_555 = np.nanmean(image_format(f['/HDFEOS/GRIDS/555nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        
        i_660= np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        qs_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        us_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        qm_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/Q_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        um_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/U_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        vza_660 = np.nanmean(image_format(f['/HDFEOS/GRIDS/660nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])
        
        i_865= np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/I/'][:])[row_lower:row_upper,column_lower:column_upper])
        qs_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        us_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_scatter/'][:])[row_lower:row_upper,column_lower:column_upper])
        qm_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/Q_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        um_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/U_meridian/'][:])[row_lower:row_upper,column_lower:column_upper])
        vaz_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_azimuth/'][:])[row_lower:row_upper,column_lower:column_upper])
        vza_865 = np.nanmean(image_format(f['/HDFEOS/GRIDS/865nm_band/Data Fields/View_zenith/'][:])[row_lower:row_upper,column_lower:column_upper])

        print(idx,i_355,i_380,i_470,i_555,i_660,i_865)
        
#________________________Section 2: Geometry Reconciliation___________________________#

        #Input Stokes Parameters
        #470
        stokesin4 = np.array([[qm_470], [um_470]]) #Meridian
        stokesin4s = np.array([[qs_470], [us_470]]) #Scattering
        #660
        stokesin6 = np.array([[qm_660], [um_660]]) #Meridian
        stokesin6s = np.array([[qs_660], [us_660]]) #Scattering
        #865
        stokesin8 = np.array([[qm_865], [um_865]]) #Meridian
        stokesin8s = np.array([[qs_865], [us_865]]) #Scattering
        

        if pol_ref_plane == 'Scattering':
            qg_470, ug_470 = stokesin4s
            qg_660, ug_660 = stokesin6s
            qg_865, ug_865 = stokesin8s
        elif pol_ref_plane == 'Meridian':
            qg_470, ug_470 = stokesin4
            qg_660, ug_660 = stokesin6
            qg_865, ug_865 = stokesin8

        if saz >= 180: 
            saz = saz - 180
        else:
            saz = saz + 180

        
        raz_355 = saz - vaz_355
        raz_380 = saz - vaz_380
        raz_445 = saz - vaz_445
        raz_470 = saz - vaz_470
        raz_555 = saz - vaz_555
        raz_660 = saz - vaz_660
        raz_865 = saz - vaz_865
        
        if raz_355 < 0:
            raz_355 = raz_355 + 360
        if raz_380 < 0:
            raz_380 = raz_380 + 360
        if raz_445 < 0:
            raz_445 = raz_445 + 360
        if raz_470 < 0:
            raz_470 = raz_470 + 360
        if raz_555 < 0:
            raz_555 = raz_555 + 360
        if raz_660 < 0:
            raz_660 = raz_660 + 360
        if raz_865 < 0:
            raz_865 = raz_865 + 360        
      
        
## NORMALIZE THE RADIANCES TO THE median EARTH-SUN DISTANCE AND CONVERT TO 
### EQUIVALENT REFLECTANCES = PI*L/E0

        eqr_i_355 = np.pi*i_355*esd**2/E0_355
        eqr_i_380 = np.pi*i_380*esd**2/E0_380
        eqr_i_445 = np.pi*i_445*esd**2/E0_445
        eqr_i_470 = np.pi*i_470*esd**2/E0_470
        eqr_i_555 = np.pi*i_555*esd**2/E0_555
        eqr_i_660 = np.pi*i_660*esd**2/E0_660
        eqr_i_865 = np.pi*i_865*esd**2/E0_865
        
        eqr_qg_470 = np.pi*qg_470*esd**2/E0_470
        eqr_qg_660 = np.pi*qg_660*esd**2/E0_660
        eqr_qg_865 = np.pi*qg_865*esd**2/E0_865
        
        eqr_ug_470 = np.pi*ug_470*esd**2/E0_470
        eqr_ug_660 = np.pi*ug_660*esd**2/E0_660
        eqr_ug_865 = np.pi*ug_865*esd**2/E0_865
                

#____________________________STORE THE DATA____________________________#


        i_median[idx,0] = eqr_i_355
        i_median[idx,1] = eqr_i_380
        i_median[idx,2] = eqr_i_445
        i_median[idx,3] = eqr_i_470
        i_median[idx,4] = eqr_i_555
        i_median[idx,5] = eqr_i_660
        i_median[idx,6] = eqr_i_865
                
        
        
        vza_median[idx,0] = vza_355
        vza_median[idx,1] = vza_380
        vza_median[idx,2] = vza_445
        vza_median[idx,3] = vza_470
        vza_median[idx,4] = vza_470
        vza_median[idx,5] = vza_470
        vza_median[idx,6] = vza_555
        vza_median[idx,7] = vza_660
        vza_median[idx,8] = vza_660
        vza_median[idx,9] = vza_660
        vza_median[idx,10] = vza_865
        vza_median[idx,11] = vza_865
        vza_median[idx,12] = vza_865
        
        raz_median[idx,0] = raz_355
        raz_median[idx,1] = raz_380
        raz_median[idx,2] = raz_445
        raz_median[idx,3] = raz_470
        raz_median[idx,4] = raz_470
        raz_median[idx,5] = raz_470
        raz_median[idx,6] = raz_555
        raz_median[idx,7] = raz_660
        raz_median[idx,8] = raz_660
        raz_median[idx,9] = raz_660
        raz_median[idx,10] = raz_865
        raz_median[idx,11] = raz_865
        raz_median[idx,12] = raz_865

        if pol_ref_plane == 'Scattering':
            q_median[idx,0] = -eqr_qg_470
            q_median[idx,1] = -eqr_qg_660
            q_median[idx,2] = -eqr_qg_865
    
            u_median[idx,0] = -eqr_ug_470
            u_median[idx,1] = -eqr_ug_660
            u_median[idx,2] = -eqr_ug_865
            
        elif pol_ref_plane == 'Meridian':
            q_median[idx,0] = eqr_qg_470
            q_median[idx,1] = eqr_qg_660
            q_median[idx,2] = eqr_qg_865
    
            u_median[idx,0] = eqr_ug_470
            u_median[idx,1] = eqr_ug_660
            u_median[idx,2] = eqr_ug_865
        

        sza_median[idx] = sza


# #__________________Section 3: Output Data in GRASP SDATA Format__________________#
# Guide to output file names
# NOTE: The options more or less correspond to GRASP retrieval.regime_of_measurement_fitting
#       0 = .radiance (option 1)
#       1-5 = .polarization (option as given)

# Change to the output directory
os.chdir(outpath) 
    
# Generate the base output file name
    #outfile_base = "AirMSPI_"+this_date_str+"_"+this_time_str+"_"
    #outfile_base = outfile_base+this_target_str+"_"
outfile_base = 'R1-Rotfrom'+pol_ref_plane

# Get the software version number to help track issues
hold = os.path.basename(__file__)
name = hold.split('_')
temp = name[len(name)-1]  # Choose the last element
hold = temp.split('.')
vers = hold[0]

### OUTPUT FILE: I IN SPECTRAL BANDS AND  I, Q, U IN POLARIZED BANDS
num_intensity = 7
num_polar = 3
num_all = num_intensity+num_polar
        
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
    
# Write out the data header line

out_str = '  1   '+date+'T'+time
out_str = out_str+'       70000.00   0   1   : NPIXELS  TIMESTAMP  HEIGHT_OBS(m)  NSURF  IFGAS    1\n'
outputFile.write(out_str)

# Generate content for sdat (single line)

out_str = '           1'  # x-coordinate (ix)
out_str = out_str+'           1'  # y-coordinate (iy)
out_str = out_str+'           1'  # Cloud Flag (0=cloud, 1=clear)
out_str = out_str+'           1'  # Pixel column in grid (icol)
out_str = out_str+'           1'  # Pixel line in grid (row)

out_str = out_str+'{:19.8f}'.format(lon)  # Longitude
out_str = out_str+'{:18.8f}'.format(lat)  # Latitude
out_str = out_str+'{:17.8f}'.format(elev) # Elevation

out_str = out_str+'      100.000000'  # Percent of land
out_str = out_str+'{:16d}'.format(num_intensity)  # Number of wavelengths (nwl)
    
  ## SET UP THE WAVELENGTH AND MEASUREMENT INFORMATION
    
# Loop through wavelengths

for loop in range(num_intensity):
    out_str = out_str+'{:17.9f}'.format(center_wave[loop]/1000.)  # Wavelengths in microns
   
# Loop over the number of types of measurements per wavelength

# for loop in range(num_intensity):
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(1) # 1 measurement per wavelength
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(3)
out_str = out_str+'{:12d}'.format(1)
out_str = out_str+'{:12d}'.format(3)
out_str = out_str+'{:12d}'.format(3)

# Loop over the measurement types per wavelength
# NOTE: Values can be found in the GRASP documentation in Table 4.5
#       41 = Normalized radiance (I = rad*pi/E0) - GRASP calls normalized (reduced) radiance

#for loop in range(num_intensity):
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)
out_str = out_str+'{:12d}'.format(41)
out_str = out_str+'{:12d}'.format(42)
out_str = out_str+'{:12d}'.format(43)


    
# Loop over the number of measurements per wavelength
# Note: This is the number of stares in the step-and-stare sequence

out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)
out_str = out_str+'{:12d}'.format(num_step)

## ANGLE DEFINITIONS

# Solar zenith angle per wavelength
# NOTE: This is per wavelength rather than per measurement (probably because of 
#       AERONET), so we take the average solar zenith angle, although this
#       varies from measurement to measurement from AirMSPI

sza_median = np.median(sza_median)

for loop in range(num_intensity):
    out_str = out_str+'{:16.8f}'.format(sza_median)

for outer in range(num_meas):
    for inner in range(num_step): 
        out_str = out_str+'{:16.8f}'.format(vza_median[inner,outer])


# Relative azimuth angle per measurement per wavelength
for outer in range(num_meas):
    for inner in range(num_step): 
        out_str = out_str+'{:16.8f}'.format(raz_median[inner,outer])


#Measurements
for outer in [0,1,2]:  # Loop over wavelengths
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])

for outer in [3]:  # Loop over wavelengths
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
    # for inner in range(num_step):  # Loop over measurements
    #     out_str = out_str+'{:16.8f}'.format(i_in_polar_median[inner,0])  # Ipol
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(q_median[inner,0])  # Q
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(u_median[inner,0])  # U

for outer in [4]:  # Loop over wavelengths
    for inner in range(num_step):  # Loop over measurements
            out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])

for outer in [5]:  # Loop over wavelengths
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(q_median[inner,1])  # Q
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(u_median[inner,1])  # U


for outer in [6]:  # Loop over wavelengths
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(i_median[inner,outer])  # I
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(q_median[inner,2])  # Q
    for inner in range(num_step):  # Loop over measurements
        out_str = out_str+'{:16.8f}'.format(u_median[inner,2])  # U
        
        
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


