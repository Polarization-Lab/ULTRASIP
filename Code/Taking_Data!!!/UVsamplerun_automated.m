% Automate Data Collection 
%Define starting directory 
saving_dir = 'C:\ULTRASIP\';

%Create data directory 
date=datestr(now,'yyyy-mm-dd');

% create a directory if it does not yet exist
% if exist(date_dir,'dir') ~= 7
%     mkdir(date_dir);
% else
% end

% Get time 
time = datestr(datetime('now', 'TimeZone', 'local'),'HHMM');

%User input 
prompt = 'Altitude? ';
altitude = input(prompt);

prompt = 'Azimuth? ';
azimuth = input(prompt);


filename = [saving_dir '' date '_' time '_' num2str(azimuth) '_' num2str(altitude) '.h5'];

usernotes = 'Taken by James Heath. Clear sky measurements for polarization characterization';

% Polarizer angles 
Degree = [0 45 90 135];

%Set trigger config of camera to manual
framesPerTrigger = 3;
triggerconfig(vid, 'manual');

%Get darkfield image 
% start(vid)
% darkfield = UV_data(vid,framesPerTrigger);
% stop(vid)
% figure;imagesc(squeeze(darkfield(1,:,:)));colorbar; axis off;
% max((darkfield(:)))

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
for ii = 1:4
%     tic   
    fprintf('Taking data %0.3f Degrees\n',Degree(ii));  
    
    %different loop values ^    
    query(ELL14,'0gp') %ask position

    image(ii,:,:) = UV_data(vid,framesPerTrigger); %take picture
    %pause(2);
    fprintf(ELL14,'0fw');
    disp('image taken')
    
%     eachtime(ii) = toc;
end

finaltime = toc;
pause(1)
query(ELL14,'0ho1')

%Close instruments
fclose(ELL14);
stop(vid)
disp('Done')


%test variables
finaltime = 1.8;
% Save data
%save('Darkfield.mat','Darkfield');
%save('clouds_1907_301603_3380118.mat','image');
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);

%write attibutes to directory
h5writeatt(filename,'/measurement/images/','meas_time', finaltime);
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','altitude', altitude);
h5writeatt(filename,'/measurement/images/','azimuth', azimuth);
h5writeatt(filename,'/measurement/images/','user_notes', usernotes);
 
% save(fullfile(date_dir,[num2str(time) '_' num2str(altitude) '_' num2str(azimuth) '.mat']), 'image');
% save(['clouds' '_' num2str(time) '_' num2str(altitude) '_' num2str(azimuth) '.mat'], 'image');