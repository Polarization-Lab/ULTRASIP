% TestIntegratingSphere.m -- Take integrating sphere measurements with
% offset of 0 as well as offset of some arbitrary delta.
%
% Written by Atkin Hyatt 08/06/2021
% Last modified by Atkin Hyatt 09/01/2021

%% Initialize
%initializeUV;
saving_dir = 'C:\ULTRASIP_Data\Data2021\Uncorrected Data\';

stop(vid)
expo = input('Exposure time in seconds? ');
src.ExposureTime = expo;

if expo >= 5
   vid.Timeout = 2 * expo;
end

iter = input('How many data points? ');

% Configure camera
triggerconfig(vid, 'manual');

%% Take data
% Start camera comms
stop(vid)
start(vid);

% initialize variables
deg = [0, 45, 90, 135];
delta = 180/(iter-1); lenDel = length(delta);
h = home;

L = length(deg);
image = zeros(iter*L,512,512);

% Collect Data
for ii = 1 : iter
    fprintf("\nData point %d\n", ii);
    
    % Scan
    for N = 1 : L
        Move_motor(deg(N),ELL14);

        pause(2)
    
        image(N+4*ii-4,:,:) = UV_data(vid,framesPerTrigger) - dark;
        disp('image taken')
    end
    
    % "Tilt" instrument by delta degrees
    if ~isempty(instrfind(ELL14,'Status','closed'))
        fopen(ELL14);
    end
    
    h = dec2hex(round(398.222222222222 * (delta + TranslateELL14(h))), 8);
    fprintf(ELL14, "%s", "0so" + h);
end

% Close instruments, reset home position
stop(vid)
fprintf(ELL14, "%s", "0so" + home);
fclose(ELL14);

%% Save data
% Define saving  directory 

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
num = int2str(iter);

global filename; global file;
file = [date '_' time '_' ex '_' num '.h5'];
filename = [saving_dir '' file];

% Create h5
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);

% Write attibutes to directory
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','datapoints', iter);
h5writeatt(filename,'/measurement/images/','exposure', expo);

%% Process data
addpath('C:\ULTRASIP\Code\Data_Analysis');
ULTRASIP_IntegratingSphere_Analysis;

%% Plot data
ReadUV_Data(file)