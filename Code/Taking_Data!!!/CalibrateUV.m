% CalibrateUV -- Scan over desired angles of interest for a new home
% postion.  Use when any polarizing elements are added/subtracted from
% setup or when 
%
% Written by Atkin Hyatt 07/09/2021
% Last modified by Atkin Hyatt 07/11/2021

function CalibrateUV(ELL14, dmin, dmax, del, count, src, vid, framesPerTrigger, plotSize)
deg = dmin : del : dmax; L = length(deg);
avcounts = zeros(1, L); maxRes = 0.01;

if del == maxRes / 10   % exit recursion
    home = dmin + 10*del;
    hexhome = dec2hex(round(home * 398.22222222),8);
    fprintf('Done\nSet home to %0.6f or %s\n', home, hexhome)
else   % continue recursion
    fprintf('Scanning for del = %0.2f\n\n', del);
    
    % set up measurement
    Move_motor(dmin,ELL14);
    fopen(ELL14); fprintf(ELL14,'%s', "0sj" + dec2hex(round(398.2222222 * del),8));
    
    % take data
    for N = 1 : L
  
        % check fan
        if src.SensorCoolerFan ~= 'on'
            src.SensorCoolerFan = 'on';
        end
        
        %for M = 1 : 2
            image = UV_data(vid,framesPerTrigger); %take picture
        %end
        fprintf('Image taken\n')
    
        avcounts(N) = mean(mean(image));  % compensate for noise in image, treat as one pixel
        fprintf('%f\n\n', avcounts(N))
        
        if deg(N) ~= dmax
            % move forward by del
            fprintf(ELL14,'0fw');
            hex = query(ELL14, "0gp");
            fprintf("Actual Position: %0.6f degrees\n", TranslateELL14(hex, 398.222222222222222));
        end
    end
    fclose(ELL14);
    
    % plot data
    subplot(plotSize, plotSize, count)
    plot(deg, avcounts); title('Average Counts vs Polarizer Angle')
    xlabel('Angle Relative to Previous Home (deg)'); ylabel('Average Counts')
    
    % find max
    maxCounts = max(avcounts);
    for N = 1 : length(avcounts)
        if avcounts(N) == maxCounts
            break 
        end
    end
    fprintf('Max counts at %f degrees\n', deg(N))
    
    % start next iteration
    count = count + 1; dmin = deg(N) - del; dmax = deg(N) + del;
    del = del / 10;
    CalibrateUV(ELL14, dmin, dmax, del, count, src, vid, framesPerTrigger, plotSize);
end
end
