# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 14:54:10 2023

@author: Clarissa
"""

#_______________Import Packages_________________#
import glob
import h5py
import numpy as np
import os
import time
import Read_AirMSPIData as r

def main():
#_______________Load in Data___________________#
# AirMSPI Step and Stare .hdf files can be downloaded from 
# https://asdc.larc.nasa.gov/data/AirMSPI/

#Work Computer
    #datapath = "C:/Users/ULTRASIP_1/Documents/Prescott817_Data"
    #outpath = "C:/Users/ULTRASIP_1/Documents/ULTRASIP/AirMSPI_FIREXAQ/Retrieval_1_121922"

#Home Computer 
    datapath = "C:/Users/Clarissa/Documents/AirMSPI/Prescott/FIREX-AQ_8172019"
    #outpath = "C:/Users/Clarissa/Documents/GitHub/ULTRASIP/AirMSPI_FIREXAQ/SDATA_Files"

    num_step = 5
    sequence_num = 0
    num_int = 7
    num_pol = 3
    
    [esd, evel_coord,lat_coord,long_coord,i,view_zen,view_az,E0_values,ipol, qm,um,dolp] = r.read_data(datapath,num_step,sequence_num,num_int,num_pol)
    
    return i,view_zen,view_az,ipol, qm,um,dolp
### END MAIN FUNCTION
if __name__ == '__main__':
    [i,view_zen,view_az,ipol, qm,um,dolp]=main() 