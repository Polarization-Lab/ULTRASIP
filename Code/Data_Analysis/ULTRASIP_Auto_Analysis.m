%% Written by Clarissa M. DeLeon - June 23rd, 2021
%% ULTRASIP_Auto_Analysis
%% Automatic Analysis: This code automatically runs through the collected 
%% data and calls ULTRASIP_Calculations to perform non-uniformity correction, and 
%% calculate stokes parameters, DoLP and AoLP
clear all; close all;

%For data after May 5th, 2021 for older data see UVanalysis in legacy code
%*******CHANGE FILENAME, FOR MULTIPLE PUT 'common_name*.h5'**************
global filename 
filename = '2021-06-23_1749_0.1.h5';

disp('starting analysis')

ULTRASIP_Calculations; %Run calculations 

%Save Calculations
%total image matrix
image = [img0 img45 img90 img135];
%total stokes matrix
stokes = [S0 s1 s2];
%matrix for AoLP and DoLP
linearpol  = [DoLP AoLP];

disp('saving...')
%create new branch for calculated data
h5create(filename,'/measurement/polarization/radiometric',size(image),"Chunksize",[323 275]);
h5create(filename,'/measurement/polarization/stokes',size(stokes),"Chunksize",[323 275]);
h5create(filename,'/measurement/polarization/polarizationmetric',size(linearpol),"Chunksize",[323 275]);

%write data to branch
h5write(filename,'/measurement/polarization/radiometric/',image);
h5write(filename,'/measurement/polarization/stokes/',stokes);
h5write(filename,'/measurement/polarization/polarizationmetric/',linearpol);

%save metadata--work in progress --CMD :S
% h5writeatt(filename,'/measurement/images/','meas_time', finaltime);
% h5writeatt(filename,'/measurement/images/','date', date);
% h5writeatt(filename,'/measurement/images/','time', time);
% h5writeatt(filename,'/measurement/images/','altitude', altitude);
% h5writeatt(filename,'/measurement/images/','azimuth', azimuth);
% h5writeatt(filename,'/measurement/images/','user_notes', usernotes);
disp('saved')
