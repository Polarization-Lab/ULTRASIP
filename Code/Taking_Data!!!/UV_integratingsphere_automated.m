% Automate Data Collection 

% Define starting directory 
saving_dir = 'C:\ULTRASIP_Data\June2021\FixExposure\';

%Create data directory 
date = datestr(now,'yyyy-mm-dd');

% Get time 
time = datestr(datetime('now', 'TimeZone', 'local'),'HHMM');

%User input
prompt = 'Delta? ';
delta = input(prompt);

% make delta a string
del = char(string(delta));
for N = 1 : length(del)
   if del(N) == '.'
       del(N) = '-';
       break
   end
end

filename = [saving_dir '' date '_' time '_' del '.h5'];

usernotes = 'Taken by Atkin Hyatt. Test measurement with the integrating sphere and fiber optic light';

% Polarizer angles 
Degree = [0, delta, 45, 45 + delta, 90, 90 + delta, 135, 135 + delta];

%Set trigger config of camera to manual
framesPerTrigger = 3;
triggerconfig(vid, 'manual');

%Take data 
disp('Taking data')
triggerconfig(vid, 'manual');

%Start camera comms
start(vid)

%Check if ELL14 is open
if ~isempty(instrfind(ELL14,'Status','close'))
    fopen(ELL14);
end

%Take images
tic
for ii = 1 : length(Degree)
    fprintf('Taking data %0.3f degrees\n',Degree(ii));
    Move_motor(Degree(ii),ELL14);

    image(ii,:,:) = UV_data(vid,framesPerTrigger); %take picture

    fprintf('\nImage taken\n\n')
    
    if ~isempty(instrfind(ELL14,'Status','close'))
        fopen(ELL14);
    end
end

finaltime = toc;
pause(1)
query(ELL14,'0ho1');

%Close instruments
fclose(ELL14);
stop(vid)
disp('Done')

%test variables
disp(finaltime)
disp(filename)
% Save data
%save('Darkfield.mat','Darkfield');
%save('clouds_1907_301603_3380118.mat','image');
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);

%write attibutes to directory
h5writeatt(filename,'/measurement/images/','meas_time', finaltime);
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','delta', delta);
h5writeatt(filename,'/measurement/images/','user_notes', usernotes);

% save(fullfile(date_dir,[num2str(time) '_' num2str(altitude) '_' num2str(azimuth) '.mat']), 'image');
% save(['clouds' '_' num2str(time) '_' num2str(altitude) '_' num2str(azimuth) '.mat'], 'image');