# ULTRASIP

***POSTER FOR GRASP WORKSHOP 2023: https://github.com/Polarization-Lab/ULTRASIP/files/11539729/DeLeon_GRASPWorkshopPoster.pdf

The UV Polarimetry for Wildfire Smoke project is combines radiative transfer simulations and development of a ultraviolet linear stokes imaging polarimeter. 
ULTRASIP is an Ultraviolet Linear Stokes Imaging Polarimeter developed by the Polarization Lab at the Wyant College of Optical Sciences at the University of Arizona. ULTRASIP was initially developed for imaging optically thin clouds but is currently being upgraded to provide ground-based remote sensing measurements of wildfire smoke.
Code for the calibration and control of ULTRASIP can be found under the Instrument Control folder. 
The radiative transfer simulations and retrievals of the microphysical properties of measured aerosols are performed using the Generalized Retrieval of Aerosol and Surface Properties (GRASP, https://www.grasp-open.com/). The code and files used for this work is found under the GRASP folder. 

First analysis of wildfire smoke is in progress using NASA data from the Air Multiangle Spectropolarimetric Imager (AirMSPI) during its depolyment during the
Joint NASA / NOAA Fire Influence on Regional to Global Environments and Air Quality (FIREX-AQ) Campaign (https://asdc.larc.nasa.gov/data/AirMSPI/FIREX-AQ/Terrain-projected_Georegistered_Radiance_Data_6/2019/08/). The data collected from AirMSPI during FIREX-AQ is first processed to read from the hdf file then sorted into subsets of data
based on location, date, and time. From these measurements, aerosol retrievals will be performed to determine the microphysical properties of the imaged smoke. Correlation between these microphysical proerties and the smokes optical
properties will be determined...to be continued....

More information found on the wiki.

Credits: Parts used from UV Mueller Matrix Polarimeter designed by Brian Daughtey and adapted by Lisa Li. ULTRASIP designed and developed by Jake Heath. Matlab software adapted by Jake Heath, Clarissa M. DeLeon, Kira Hart Shanks, and Meredith Kupinski. Part collection and lab assitance by Jeremy Parkinson.
