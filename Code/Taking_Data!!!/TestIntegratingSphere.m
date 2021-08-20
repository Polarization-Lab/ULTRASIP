% TestIntegratingSphere.m -- Take integrating sphere measurements with
% offset of 0 as well as offset of some arbitrary delta.
%
% Written by Atkin Hyatt 08/06/2021
% Last modified by Atkin Hyatt 08/20/2021

%% Initialize
%initializeUV;
stop(vid)
expo = input('Exposure time in seconds? ');
src.ExposureTime = expo;

if expo >= 5
   vid.Timeout = 2 * expo;
end

delta = input('Delta value in degrees? ');

% Configure camera
triggerconfig(vid, 'manual');

%% Take data
% Start camera comms
start(vid);

% initialize variables
deg = [0, 45, 90, 135];
L = length(deg);
image = zeros(2*L,512,512);

% First data point
for N = 1 : L
    Move_motor(deg(N),ELL14);

    pause(2)
    
    image(N,:,:) = UV_data(vid,framesPerTrigger) - dark;
    disp('image taken')
end

% "Tilt" instrument to delta
fopen(ELL14);
h = dec2hex(round(398.222222222222 * delta + hex2dec(home)), 8);
trueDel = TranslateELL14(query(ELL14, "0go"));
fprintf(ELL14, "%s", "0so" + h);
fclose(ELL14);

% Second data point
for N = 1 : L
    Move_motor(deg(N),ELL14);
    
    pause(2)
    
    image(N + L,:,:) = UV_data(vid,framesPerTrigger) - dark;
    disp('image taken')
end

%Close instruments
stop(vid)
fopen(ELL14);
fprintf(ELL14, "%s", "0so" + home);
fclose(ELL14);

%% Save data
% Define saving  directory 
saving_dir = 'C:\ULTRASIP_Data\July2021\Uncorrected Data\';

% Create data directory 
date = datestr(now,'yyyy-mm-dd'); % get date
time = datestr(datetime('now', 'TimeZone', 'local'),'HHMM'); % get time

% Convert decimals to dashes
ex = char(string(expo));
for N = 1 : length(ex)
   if ex(N) == '.'
       ex(N) = '-';
       break
   end
end

del = char(string(delta));
for N = 1 : length(del)
   if del(N) == '.'
       del(N) = '-';
       break
   end
end

global filename; global file;
file = [date '_' time '_' ex '_' del '.h5'];
filename = [saving_dir '' file];

% Create h5
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);

% Write attibutes to directory
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','delta', trueDel);
h5writeatt(filename,'/measurement/images/','exposure', expo);

%% Process data
addpath('C:\ULTRASIP\Code\Data_Analysis');
ULTRASIP_IntegratingSphere_Analysis;

%% Plot data
ReadUV_Data(f, trueDel)