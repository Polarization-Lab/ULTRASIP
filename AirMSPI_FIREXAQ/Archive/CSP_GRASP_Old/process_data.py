# -*- coding: utf-8 -*-
"""
@author: C.M.DeLeon
This script is to read in the hdf AirMSPI files then create a processed dataset
selecting only data types of interest
"""
#Last edit: 06.19.2022

#Import Libraries 
import pickle 
import numpy as np
import h5py
import os 
import glob
from ypstruct import struct 

struct.__getstate__ = lambda self: self.__dict__
struct.__setstate__ = lambda self, x: self.__dict__.update(x)


#Change to file directory !!!
path = "D:\AirMSPI_FIREXAQ\Arizona"
os.chdir(path)

#Make list of file names 
filenames = glob.glob("AirMSPI*.hdf")

index = len(filenames)

for i in range(0,index):
    #Basic metadata
    date = filenames[i][28:30] + "." + filenames[i][30:32] +"."+ filenames[i][24:28]
    time = filenames[i][33:35] + ":" + filenames[i][35:37] +":"+ filenames[i][37:40]
    savefilename = "Processed_" + filenames[i][0:66] + ".pickle"

    #Extract data and save new processed data
    processing_file = h5py.File(filenames[i],'r+')


    #Define struct
    Processed = struct();

    #Location Information
    Processed.latitude = processing_file["/HDFEOS/GRIDS/Ancillary/Data Fields/Latitude"][()]
    Processed.longitude = processing_file["/HDFEOS/GRIDS/Ancillary/Data Fields/Longitude"][()]


    #Intensities 
    Processed.Intensity355 = processing_file["/HDFEOS/GRIDS/355nm_band/Data Fields/I"][()]
    Processed.Intensity380 = processing_file["/HDFEOS/GRIDS/380nm_band/Data Fields/I"][()]
    Processed.Intensity445 = processing_file["/HDFEOS/GRIDS/445nm_band/Data Fields/I"][()]
    Processed.Intensity470 = processing_file["/HDFEOS/GRIDS/470nm_band/Data Fields/I"][()]
    Processed.Intensity555 = processing_file["/HDFEOS/GRIDS/555nm_band/Data Fields/I"][()]
    Processed.Intensity660 = processing_file["/HDFEOS/GRIDS/660nm_band/Data Fields/I"][()]
    Processed.Intensity865 = processing_file["/HDFEOS/GRIDS/865nm_band/Data Fields/I"][()]
    Processed.Intensity935 = processing_file["/HDFEOS/GRIDS/935nm_band/Data Fields/I"][()]

    #IntensityMask
    Processed.IntensityMask355 = processing_file["/HDFEOS/GRIDS/355nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask380 = processing_file["/HDFEOS/GRIDS/380nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask445 = processing_file["/HDFEOS/GRIDS/445nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask470 = processing_file["/HDFEOS/GRIDS/470nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask555 = processing_file["/HDFEOS/GRIDS/555nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask660 = processing_file["/HDFEOS/GRIDS/660nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask865 = processing_file["/HDFEOS/GRIDS/865nm_band/Data Fields/I.mask"][()]
    Processed.IntensityMask935 = processing_file["/HDFEOS/GRIDS/935nm_band/Data Fields/I.mask"][()]


    #DoLP for each channel 
    Processed.dolp470 = processing_file["/HDFEOS/GRIDS/470nm_band/Data Fields/DOLP"][()]
    Processed.dolp660 = processing_file["/HDFEOS/GRIDS/660nm_band/Data Fields/DOLP"][()]
    Processed.dolp865 = processing_file["/HDFEOS/GRIDS/865nm_band/Data Fields/DOLP"][()]

 
    #Save processed structure
    saveit = open(savefilename,'ab') #creates file if it doesnt exist and saves in binary
    pickle.dump(Processed, saveit)          

