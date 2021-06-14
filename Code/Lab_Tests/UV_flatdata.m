%This Script is to obtain a flat field image 
%Take an image of a relatively uniform object e.g. a blank sheet of copy
%paper
%with nothing in front of the camera 

function image = UV_flatdata(vid,framesPerTrigger,darkfield)
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
Flat_raw = double(im);

for i = 1:100
    Flat_cal(i,:,:) = squeeze(Flat_raw.darkfield(i,:,:)) - squeeze(darkfield.darkfield(1,:,:));
    
    Flat_cal(:,:,i) = squeeze(Flat_raw.darkfield(:,:,i)) - squeeze(darkfield.darkfield(:,:,i));


end

Flat_avg = Flat_cal(1,:,:);

for i = 2:100
    Flat_avg = Flat_avg + Flat_cal(i,:,:);
end

Flat_avg = Flat_avg./100;
Flat_Master = Flat_avg./mean(Flat_avg);

image = Flat_Master;

% 
% image = im(1,:,:);
% 
% for i = 2:framesPerTrigger
%     image = image + im(i,:,:);
% end
% 
% image = image/framesPerTrigger;
% 
