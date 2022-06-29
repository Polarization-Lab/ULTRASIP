%Take camera image 
%Written by Clarissa DeLeon 1.19.2022 

% Use this function to take image with Hamamatsu camera 
%Camera details:  Hamamatsu EM-CCD Digital Camera ImagEM 
    % MODEL: C9100-13 Serial No. 1Y0726 

%Funtion in/outputs:

function[image] = Take_image_ULTRASIP(vid,framesPerTrigger,exposure)

src.ExposureTime = exposure;

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

end