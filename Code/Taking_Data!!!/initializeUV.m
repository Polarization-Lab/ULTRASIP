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
%comPort = 'COM1'; %Whichever port the ESP301 is plugged in
%esp301 = espConnect(comPort);

%% Connect to piezo Motor
disp('Connecting ELL14')
comPort = 'COM1'; %Whichever port the ESP301 is plugged in

% Set home position
global home
home = 'FFFF9C72';

ELL14 = ELL14Connect(comPort, home);
disp('ELL14 connected')

%% Connect to Camera 
%Initialization
mode = 1; % 1x1 binning
%exposureTime = 6; % in seconds
framesPerTrigger = 1;

vid = CameraConnect(mode,framesPerTrigger);
src = getselectedsource(vid);

src.TriggerConnector = 'bnc';

%ensure that fan is ON
src.SensorCoolerFan = 'on';

% Change exposure after initialize

src.ExposureTime = 0.3; %Exposure time of Camera 
%if exposureTime >= 5
%    vid.Timeout = 2 * exposureTime;
%end