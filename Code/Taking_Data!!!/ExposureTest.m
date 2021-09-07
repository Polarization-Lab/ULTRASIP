% ExposureTest -- Take picture of setup with varying exposures and plot counts as
% a function of exposure time in seconds.  NOTE: min exposure time is 30 ms
%
% Written by Atkin Hyatt 07/01/2021
% Last modified by Atkin Hyatt 07/19/2021

% Initialize
%initializeUV

%expo = 0.01 : 0.01 : 0.1; L = length(expo);
expo = 0.5;
avcounts0 = zeros(1,L);
avcounts90 = zeros(1,L);
image = zeros(1,512,512);

% 0 deg
Move_motor(0, ELL14); fprintf('\n\n0 degrees\n')
for N = 1 : L
    fprintf('\n%f\n',expo(N))
    src.ExposureTime = expo(N);
    fprintf('Real - %f\n',src.ExposureTime)
   
    image = UV_data(vid,framesPerTrigger); % take picture
    
    fprintf('Image taken\n')
    
    avcounts0(N) = mean(mean(image));  % compensate for noise in image, treat as one pixel
    fprintf('%f\n',avcounts0(N))
    clear src.ExposureTime
end
plot(expo, avcounts0);
legend({'0 deg', '90 deg'}, 'FontSize', 14)
title('Average Counts vs Exposure Time'); xlabel('Exposure Time (sec)'); ylabel('Average Counts');
hold on

% 90 deg
Move_motor(90,ELL14); fprintf('\n\n90 degrees\n')
for N = 1 : L
    fprintf('\n%f\n',expo(N))
    src.ExposureTime = expo(N);
    fprintf('Real - %f\n',src.ExposureTime)
   
    image = UV_data(vid,framesPerTrigger); % take picture
    
    fprintf('Image taken\n')
    
    avcounts90(N) = mean(mean(image));  % compensate for noise in image, treat as one pixel
    fprintf('%f\n',avcounts90(N))
    clear src.ExposureTime
end
plot(expo, avcounts90);
legend({'0 deg', '90 deg'}, 'FontSize', 14)
title('Average Counts vs Exposure Time'); xlabel('Exposure Time (sec)'); ylabel('Average Counts');

% Find min exposure time (only run this part of the script to figure this
% out)
% for N = length(expo) : -1 : 1
%    disp(expo(N))
%    src.ExposureTime = expo(N);
%    disp('done')
%    clear src.ExposureTime
%    pause(0.1);
% end