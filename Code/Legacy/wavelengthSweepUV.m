function [] = wavelengthSweepUV(fn,wavelength, exposure , vid, num_meas, COMmono, esp301, framesPerTrigger)
%WAVELENGTHSWEEP Summary of this function goes here
%   fn  filename string path to .h5 fil.e
%   WAVELENGTH double wavelength of exposure [nm]
%   EXPOSURE double exposure time for this wavelength [sec]
%   PSG array with PSG positions
%   PSA array with PSA positions
%   VID video input object 
%   COMmono serial variable to monochromator 
%   COMdmm serial variable to dmm
%   xps XPS to motor 
%   framesPerTrigger number of frames per trigger

%load PSA/PSG Angles
[PSA, PSG] = generate_PSAG_angles(num_meas);
 
%set monochromator wavelength
changeWavelength(COMmono,wavelength)

%configure camera settings
src = getselectedsource(vid);
triggerconfig(vid, 'immediate');
vid.FramesPerTrigger = framesPerTrigger;
src.ExposureTimeControl = 'normal';
src.ExposureTime = exposure;

%given settings, find correct image dimension for saving
start(vid)
im = getdata(vid);
imdim = size(im);

%create wavelength fp
wavename = strcat('/images/wave',num2str(wavelength),'/'); %name for this wavelength group

%take darkfield image
NI_shutter_UV(0)%close shutter
pause(1)
[dark_im, dark_ref] = take_snapshot(vid, exposure , framesPerTrigger);
NI_shutter_UV(1) %open_shutter

pause(exposure)


%write dark image to h5
imname = strcat('/images/wave',num2str(wavelength),'/darkdata');
dmmname= strcat('/images/wave',num2str(wavelength),'/darkref');

h5create(fn,imname,size(dark_im))%create image dataset
h5create(fn,dmmname,size(dark_ref))%create image dataset

h5write(fn,imname,dark_im);
h5write(fn,dmmname, dark_ref);

%set up measurement for number of PSG/PSA measurements
i=1;

%loop through positions
while i < num_meas +1
    g = PSG(i);
    a = PSA(i);
    
    %% CHANGE
    movePSG(xps,g) %MOVE psg
    movePSA(xps,a) %MOVE PSA
    %%
    %define image group name
    meas_name = strcat(wavename,'meas',num2str(i),'/');     
    
    %take images
    [im, ref] = take_snapshot(vid, exposure , framesPerTrigger);
           
    %write to h5
    imname = strcat(meas_name,'imagedata');
    dmmname= strcat(meas_name,'refdata');

    h5create(fn,imname,size(im))%create image dataset
    h5create(fn,dmmname,size(ref))%create image dataset

    h5write(fn,imname,im);
    h5write(fn,dmmname,ref);

    clear im
    clear ref
    
    i=i+1;
end

%write attibutes to directory
 h5writeatt(fn,wavename,'exposure_time', exposure);
 h5writeatt(fn,wavename,'frames_per_trigger', framesPerTrigger);
 h5writeatt(fn,wavename,'image_dimension', imdim);
end

