%James Heath heathjam@email.arizona.edu
%Sept 25 2020
%This script initializes UV Polarimeter instruments

fclose('all')

clear all
close all
clc
instrreset %clear and reset any existing port communications

% DMM = initializeUV_DMM();

comPort = 'COM1'; %Whichever port the ESP301 is plugged in

%%Initialization
ROI =  [274 274 1500 1500];
mode = 1 ; % 1x1 binning
framespertrigger = 3;

% xps = initializeMotor();
espConnect(comPort)

vid = initializeUVCamera(mode,ROI,framespertrigger);
src = getselectedsource(vid);

src.ExposureTime = 1; %Exposure time of Camera



