% CalibrateUV -- Scan over desired angles of interest for a new home
% postion.  Use when any polarizing elements are added/subtracted from
% setup
%
% Written by Atkin Hyatt 07/07/2021
% Last modified by Atkin Hyatt 07/10/2021

maxRes = 0.01; count = 1;
dmin = input('Enter minimum angle to scan: ');
dmax = input('Enter maximum angle to scan: ');
del = input('Initial angle incriment: ');

C = del;
while C > maxRes
    C = C / 10;
    count = count + 1;
end
plotSize = sqrt(count);

CalibrateUV(ELL14, dmin, dmax, del, 1, src, vid, framesPerTrigger, plotSize)