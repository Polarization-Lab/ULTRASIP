% CalibrateUV -- Scan over desired angles of interest for a new home
% postion.  Use when any polarizing elements are added/subtracted from
% setup.  Change variable "home" in initializeUV.m
%
% Written by Atkin Hyatt 07/09/2021
% Last modified by Atkin Hyatt 07/19/2021

maxRes = 0.01; count = 1; dark = zeros(1,512,512); pic = zeros(1,512,512);
dmin = input('Enter minimum angle to scan: ');
dmax = input('Enter maximum angle to scan: ');
del = input('Initial angle incriment: ');

% determine number of subplots and total plot size
C = del;
while C > maxRes
    C = C / 10;
    count = count + 1;
end
plotSize = ceil(sqrt(count));

framesPerTrigger = 3;
triggerconfig(vid, 'manual');
start(vid)

% take dark measurement
input('Turn off source for dark measurement, enter any character to continue: ', 's');
for N = 1 : 2
    pic = UV_data(vid,framesPerTrigger);
    dark = dark + pic;
end
dark = reshape(dark, 512, 512) ./ 2;
input('Turn on source for alignement measurement, enter any character to continue: ', 's');

CalibrateUV(ELL14, dmin, dmax, del, count, src, vid, framesPerTrigger, plotSize, dark)

stop(vid)