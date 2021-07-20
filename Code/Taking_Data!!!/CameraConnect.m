function [vid] = CameraConnect(mode,FramesPerTrigger)
%INITIALIZECAMERA Summary of this function goes here
%James Heath heathjam@email.arizona.edu
%Sept. 25 2020
%This script is designed to initialize the camera
%R2020a required
%Bin size can be 1,2 or 4
% ROI POsition is [xstart,ystart,w,h]
%FramerPerTrigger is an interger

disp('Initializing Camera')
if mode == 1
    vid = videoinput('hamamatsu', 1, 'MONO16_512x512_FastMode'); %vid source for camera, 1x1 BIN
elseif mode == 2
    vid = videoinput('hamamatsu', 2, 'MONO16_512x512_FastMode'); %2x2bin
else
    disp('Invalid bin size, enter 1 or 2')
end

vid.FramesPerTrigger = FramesPerTrigger; 

disp('done')

end
