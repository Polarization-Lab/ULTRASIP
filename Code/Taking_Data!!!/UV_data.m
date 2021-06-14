function image = UV_data(vid,framesPerTrigger)
%%
% pause(3)
% src = getselectedsource(vid);
% src.ExposureTime = 0.35; %Exposure time of Camera
% tic
% start(vid)
% im = getdata(vid);
% stop(vid);
% toc
% 
% tic
% im = getsnapshot(vid);
% toc
%%
% Configure the object for manual trigger mode.

% Measure the time to acquire 20 frames.
im = zeros(framesPerTrigger,512,512);

for i = 1:framesPerTrigger
    im(i,:,:) = getsnapshot(vid);
end

%Create averaged image over framesPerTrigger
im = double(im);
image = im(1,:,:);

for i = 2:framesPerTrigger
    image = image + im(i,:,:);
end

image = image/framesPerTrigger;

