% Record.m -- measure source intensity as a function of time to measure
% source flicker
%
% Written by Atkin Hyatt 08/05/2021
% Last modified by Atkin Hyatt 08/10/2021

%initializeUV
clear im time
triggerconfig(vid, 'manual');
src.ExposureTime = 0.148;
stop(vid)
start(vid)

for i = 1 : 1000
    tic;
    image = UV_data(vid,framesPerTrigger) - dark;
    im(i) = mean(mean(image));
    toc;
    if i == 1
        time(i) = toc;
    else
        time(i) = toc + time(i - 1);
    end
    
    plot(time, im)
end
stop(vid)