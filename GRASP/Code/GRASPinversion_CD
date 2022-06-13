#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:19:44 2022

@author: cdeleon
"""

#Code for GRASP inversion

#packages
from runGRASP import graspRun, pixel
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

#paths
fwdModelYAMLpath = '/home/cdeleon/ULTRASIP/Code/GRASP/SettingFiles/settings_1lambda_airmspi_inversion.yml'
binPathGRASP = '/home/cdeleon/grasp/build/bin/grasp'
krnlPathGRASP = '/home/cdeleon/grasp/src/retrieval/internal_files'

gr = graspRun(pathYAML=fwdModelYAMLpath, releaseYAML=True, verbose=True)
gr.runGRASP(binPathGRASP=binPathGRASP, krnlPathGRASP=krnlPathGRASP)