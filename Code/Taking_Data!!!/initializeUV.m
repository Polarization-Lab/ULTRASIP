%James Heath heathjam@email.arizona.edu
%Sept 25 2020
%This script initializes UV Polarimeter instruments

%Instrument Reset
fclose('all')

clear all
close all
clc
instrreset %clear and reset any existing port communications

% DMM = initializeUV_DMM();
%% Connect to ESP Motor
comPort = 'COM1'; %Whichever port the ESP301 is plugged in
esp301 = espConnect(comPort);

%% Connect to piezo Motor
%comPort = 'COM1'; %Whichever port the ESP301 is plugged in
% xps = initializeMotor();
%ELL14 = ELL14Connect(comPort);

%% Connect to Camera 
%Initialization
mode = 1; % 1x1 binning

framesPerTrigger = 1;

vid = CameraConnect(mode,framesPerTrigger);
src = getselectedsource(vid);

src.TriggerConnector = 'bnc';
src.ExposureTime = 6; %Exposure time of Camera


