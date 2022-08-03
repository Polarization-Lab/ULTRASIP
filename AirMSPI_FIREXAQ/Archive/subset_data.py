# -*- coding: utf-8 -*-
"""
@author: C.M.DeLeon

This is a code to create a subset of AirMSPI Data by Date/Time/Location
Note: Must process data first 
"""
#Last edit: 06.19.2022

#Import Libraries
import numpy as np 
import pickle 
import h5py
import os 
import glob
from ypstruct import struct 

#define variables
struct.__getstate__ = lambda self: self.__dict__
struct.__setstate__ = lambda self, x: self.__dict__.update(x)

#Change to file directory!!!
path = "D:\AirMSPI_FIREXAQ\Arizona"
os.chdir(path)

#Load in processed data 
processedfiles = glob.glob("Processed*.pickle")
Lat = [0 for x in range(0,len(processedfiles)-1)]
Long = [0 for x in range(0,len(processedfiles)-1)]

for i in range(0,len(processedfiles)-1):

    #Data will be sorted by Latitude/Longitude then they will automatically sort 
    #into Date/Time by name convention. 
    pfile = pickle.load(open(processedfiles[i], "rb"))
    Lat[i] = np.mean(pfile.latitude)
    Long[i]= np.mean(pfile.longitude)
    
Latsorted = np.argsort(Lat)
Longsorted = np.argsort(Long)

sortedFiles = [0 for x in range(0,len(Latsorted)-1)]

#Change to save directory!!!
path = "D:\AirMSPI_FIREXAQ\Arizona\SortedData"
os.chdir(path)

for i in range(0,len(Latsorted)-1):
    
    sortedFiles[i] = processedfiles[Latsorted[i]]
    savefilename = "lat" + str(round(Lat[i],3)) + "long" + str(round(abs(Long[i]),3)) + sortedFiles[i][0:78] + ".pickle"
    pickle.dump(sortedFiles[i], open(savefilename,'ab'))
