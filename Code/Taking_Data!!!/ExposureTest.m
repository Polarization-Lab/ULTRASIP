% ExposureTest -- Take picture of setup with varying exposures and plot counts as
% a function of exposure time in seconds.
%
% Written by Atkin Hyatt 07/01/2021
% Last modified by Atkin Hyatt 09/13/2021

% Initialize
%initializeUV
stop(vid)
hold off
clear src.ExposureTime expo avcounts0

expo1 = 0.001; %L = length(expo);
expo2 = 3;
dexpo = 0.001;

image = zeros(1,512,512);
start(vid)

% 0 deg
Move_motor(0, ELL14);
for N = 1 : ((expo2-expo1)/dexpo + 1)
    expo(N) = expo1 + (dexpo*(N-1));
    
    fprintf('\n%f\n',expo(N))
    src.ExposureTime = expo(N);
    fprintf('Real - %f\n',src.ExposureTime)
   
    image = UV_data(vid,framesPerTrigger); % take picture
    fprintf('Image taken\n')
    
    avcounts0(N) = mean(mean(image));  % compensate for noise in image, treat as one pixel
    fprintf('%f\n',avcounts0(N))
    clear src.ExposureTime
    
    plot(expo, avcounts0);
    title('Average Counts vs Exposure Time');
    xlabel('Exposure Time (sec)'); ylabel('Average Counts');
end


% 90 deg
% Move_motor(90,ELL14); fprintf('\n\n90 degrees\n')
% for N = 1 : ((expo2-expo1)/dexpo + 1)
%     expo(N) = expo1 + (dexpo*(N-1));
%     
%     fprintf('\n%f\n',expo(N))
%     src.ExposureTime = expo(N);
%     fprintf('Real - %f\n',src.ExposureTime)
%    
%     image = UV_data(vid,framesPerTrigger); % take picture
%     fprintf('Image taken\n')
%     
%     avcounts0(N) = mean(mean(image));  % compensate for noise in image, treat as one pixel
%     fprintf('%f\n',avcounts0(N))
%     clear src.ExposureTime
%     
%     plot(expo, avcounts0);
%     title('Average Counts vs Exposure Time');
%     xlabel('Exposure Time (sec)'); ylabel('Average Counts');
% end
% legend({'0 deg', '90 deg'}, 'FontSize', 14)
stop(vid)
%%
% Find min exposure time (only run this part of the script to figure this
% out)
%initializeUV
expo = 0 : 0.001 : 0.05;
for N = length(expo) : -1 : 1
   disp(expo(N))
   src.ExposureTime = expo(N);
   disp('done')
   clear src.ExposureTime
   pause(0.01);
end