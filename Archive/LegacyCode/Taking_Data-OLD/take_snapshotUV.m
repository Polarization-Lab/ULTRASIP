function [image, ref] = take_snapshotUV(vid, exposure , framesPerTrigger)
%take_snapshot takes image and reference data
%   vid video


% Create the object.
%ai = daq('ni') ;

% Add one channel for recording the reference
%addinput(ai,'Dev2','ai2','Voltage');

% Set the sample rate to 1000 Hz.
%ai.Rate = 1000;
% determine how many measurements to aquire 
%ref_size = exposure * 1000;

%calculate container sizes
start(vid)
im = getdata(vid);
imdim = size(im);
image = zeros(imdim);
%ref = zeros(1,ref_size);


%aquire image simultaneously
im = getdata(vid);
%ref = read(ai, ref_size, "OutputFormat", "Matrix");

stop(vid);


%calculate average image
image = im(:,:,1);
for i = 2:framesPerTrigger
    image = image + im(:,:,i);
end
image = image/framesPerTrigger;

%calculate reference amplitude
%ref = max(ref) - min(ref);

end

