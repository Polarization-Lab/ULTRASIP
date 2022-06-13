%James Heath heathjam@email.arizona.edu
%Sept 25 2020
%This script initializes UV Polarimeter instruments
% Last modified by Atkin Hyatt 10/18/2021

%Instrument Reset
fclose('all');
clear all;
close all
clc
instrreset %clear and reset any existing port communications

% DMM = initializeUV_DMM();
%% Connect to ESP Motor
%comPort = 'COM1'; %Whichever port the ESP301 is plugged in
%esp301 = espConnect(comPort);

%% Connect to piezo Motor
disp('Connecting ELL14')
comPort = 'COM1'; %Whichever port the ELL14 is plugged in

% Set home position
global home
home = '00008C9D';

% Percent of max speed
speedPer = 100;

ELL14 = ELL14Connect(comPort, home);
fopen(ELL14);
fprintf(ELL14, "%s", "0sv" + dec2hex(speedPer));
fclose(ELL14);
disp('ELL14 connected')

%% Connect to Camera 
%Initialization
mode = 1; % 1x1 binning
%exposureTime = 6; % in seconds
framesPerTrigger = 1;

vid = CameraConnect(mode,framesPerTrigger);
src = vid.Source;

src.TriggerConnector = 'bnc';

%ensure that fan is ON
src.SensorCoolerFan = 'on';

% Change exposure after initialize
% FindExposure
% src.ExposureTime = opEx; %Exposure time of Camera

src.ExposureTime = input('Exposure Time in seconds? ');
%if exposureTime >= 5
%    vid.Timeout = 2 * exposureTime;
%end

%% Measure darkfield
fprintf("Turn off source for darkfield measurement\n")
triggerconfig(vid, 'manual');

% Get around Hamamatsu driver issue with exposure time (idk why but it works)
start(vid)
stop(vid)
start(vid)

% Countdown
fprintf("Measuring darkfield in\n")
for ii = 0 : 2
   fprintf("%d\n", 3 - ii)
   pause(1)
end

fprintf("Starting measurement...\n")
sum = zeros(1, 512, 512);
for ii = 1 : 3
    dark = UV_data(vid,framesPerTrigger); %take picture
    sum = dark + sum;
end
dark = sum ./ 3;

stop(vid); clear ii; clear sum;
fprintf("Measurement complete, camera stopped\n")
fprintf("Initialization complete\n")