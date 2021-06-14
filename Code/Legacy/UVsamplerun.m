%% Home Motors

%Check if MC is already opened
% disp('Homing')
% if ~isempty(instrfind(esp301,'Status','close'))
%     fopen(esp301);
% end
% 
% %Send home command
% fprintf(esp301,'1OR;1WS');
% 
% %close connection
% fclose(esp301);
% 
% %Wait for motor to home
% pause(15);
% 
% %Check if MC is already opened
% if ~isempty(instrfind(esp301,'Status','close'))
%     fopen(esp301);
%     
% end
% 
% %Ask motor position
% query(esp301,'1VA?')
% 
% %close connection
% fclose(esp301);
% 
% disp('Homed')

%% Darkfield
% 2 sec mountain cloud
%0.13 all else
%pause(60);
%Set trigger config of camera to manual
framesPerTrigger = 3;
triggerconfig(vid, 'manual');
start(vid)
%darkfield = getdata(vid);
darkfield = UV_data(vid,framesPerTrigger);
stop(vid)

figure;imagesc(squeeze(darkfield(1,:,:)));colorbar; axis off;
max((darkfield(:)))

% %% Flatfield
% % 2 sec mountain cloud
% %0.13 all else
% %pause(60);
% %Set trigger config of camera to manual
% framesPerTrigger = 100;
% triggerconfig(vid, 'manual');
% start(vid)
% %darkfield = getdata(vid);
% flatfield = UV_flatdata(vid,framesPerTrigger,darkfield);
% stop(vid)
% 
% imagesc(squeeze(flatfield(1,:,:)));colorbar;
% max((flatfield(:)))

%% Degree Inputs
Degree = [0 45 90 135];
%Degree = [0];
%%
triggerconfig(vid, 'manual');
disp('Taking data')

%Start camera comms

start(vid)


%Check if ESP301 is open
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

%% Save data
save('Darkfield.mat','Darkfield');
%%
save('clouds_1907_301603_3380118.mat','image');


% Jake this is how you can get UTC time,I may have to go over the code with
%you to determine how you want  it to autosave 
%dtLCL = datetime('now', 'TimeZone', 'local'); %Current local time
%dtUTC = datetime(dtLCL, 'Timezone', 'Z'); %Current UTC time 
%DateString = datestr(dtUTC); %To get UTC time and date in one string

%%
%write attibutes to directory
 h5writeatt(name,'/images/','start_time', starttime);
 h5writeatt(name,'/images/','end_time', endtime);
 h5writeatt(name,'/images/','user_notes', usernotes);

% close ports 
fclose('all');
close all
clc
instrreset
