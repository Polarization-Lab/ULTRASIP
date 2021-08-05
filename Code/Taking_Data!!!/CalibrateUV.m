% CalibrateUV -- Scan over desired angles of interest for a new home
% postion.  Use when any polarizing elements are added/subtracted from
% setup.  Change variable "home" in initializeUV.m
%
% Run Calibrate.m
%
% Written by Atkin Hyatt 07/09/2021
% Last modified by Atkin Hyatt 07/19/2021

function CalibrateUV(ELL14, dmin, dmax, del, count, src, vid, framesPerTrigger, plotSize, dark)
deg = dmin : del : dmax; L = length(deg); trueDeg = zeros(1,L); stdev = zeros(1,L);
avcounts = zeros(1, L); maxRes = 0.01;

% check resolution
if del == maxRes / 10   % exit recursion
    % get optimized home
    newhome = dmin + 10*del;
    
    % find the new offset from current home
    %offset = home + newhome;
    hexhome = dec2hex(round(newhome * 398.22222222),8);
    
    % tell user new offset
    fprintf('Done\nSet home to %0.6f or %s\n', newhome, hexhome)
    stop(vid)
else   % continue recursion
    fprintf('Scanning for del = %0.2f\n\n', del);
    
    % take data
    for N = 1 : L
        trueDeg(N) = Move_motor(deg(N), ELL14);
        % check fan
        if src.SensorCoolerFan ~= 'on'
            src.SensorCoolerFan = 'on';
        end
        
        image = zeros(1, 512, 512);
        for M = 1 : 2
            pic = UV_data(vid,framesPerTrigger); %take picture
            image = image + pic;
        end
        image = image ./ 2;
        fprintf('Image taken\n')
        
        
        im = reshape(image,512,512) - dark;
        stdev(N) = std(reshape(im, 1, 512*512));
        
        avcounts(N) = mean(mean(im));  % compensate for noise in image, treat as one pixel
        fprintf('%f\n\n', avcounts(N))
    end
    
    % plot data
    subplot(plotSize, plotSize, count)
    errorbar(trueDeg, avcounts, stdev); title('Av Counts vs Polarizer Angle')
    xlabel('Angle Relative to Current Home (deg)'); ylabel('Average Counts')
    
    % find max
    maxCounts = max(avcounts);
    for N = 1 : length(avcounts)
        if avcounts(N) == maxCounts
            break 
        end
    end
    fprintf('Max counts at %f degrees\n', deg(N))
    
    % start next iteration
    count = count - 1; dmin = deg(N) - del; dmax = deg(N) + del;
    del = del / 10;
    CalibrateUV(ELL14, dmin, dmax, del, count, src, vid, framesPerTrigger, plotSize, dark)
end
end
