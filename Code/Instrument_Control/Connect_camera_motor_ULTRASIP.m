% Connect ULTRASIP Components 
% Written by Clarissa DeLeon 1.14.2022 
% Run this script to connect to camera and motor 

%Camera details:  Hamamatsu EM-CCD Digital Camera ImagEM 
    % MODEL: C9100-13 Serial No. 1Y0726 

%Motor details: Thorlabs Rotation Mount and interface board
    %MODEL: ELL14K 

%Funtion in/outputs:
%Mode can be 1,2 or 4
%ROI Position is [xstart,ystart,w,h]
%FramerPerTrigger is an interger
%comPort for motor

    function[src] = Connect_camera_motor_ULTRASIP(mode, FramesPerTrigger, comPort)

%----------------------Camera Connection-----------------------------%

disp('Initializing Camera')

%vid source for camera, 1x1 BIN
if mode == 1
    vid = videoinput('hamamatsu', 2, 'MONO16_2048x2048_FastMode'); 

%2x2bin
elseif mode == 2
    vid = videoinput('hamamatsu', 2, 'MONO16_BIN2x2_1024x1024_FastMode'); 

%4x4bin
elseif mode == 4
    vid = videoinput('hamamatsu', 2, 'MONO16_BIN4x4_512x512_FastMode'); 
else
    disp('Invalid bin size, enter 1,2, or 4')
end

vid.FramesPerTrigger = FramesPerTrigger; 
%vid.ROIPosition = ROIPosition;

%ensure that fan is ON
src = getselectedsource(vid);
src.HighDynamicRangeMode = 'on';
src.SensorCoolerFan = 'on';


%------------------------Motor Connection------------------------%
ELL14 = serial(comPort);
set(ELL14, 'baudrate', 9600,'databits',8,'stopbits',1);

fopen(ELL14);

%set home position and home
home = '00008C9D' ; %desired hex value 
fprintf(ELL14,"%s", "0so" + home);
fprintf(ELL14, '0ho0');

end
