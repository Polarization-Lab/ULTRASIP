% Take a series of data at varying analyzer and pan/tilt degrees
% Written by Clarissa DeLeon 1.14.2022 

%Function inputs: 
% deg_range = degs to set motor to for a set of measurements
% ex: 0:45:135
%saving_dir = define where to save files 
%ex: 'C:/ULTRASIP/Jan2022'
%exposure = exposure time of camera in seconds

function [image] = Run_Measurement_ULTRASIP(deg_range,saving_dir,exposure)

%get date 
date=datestr(now,'yyyy-mm-dd');

% Get time 
time = datestr(datetime('now', 'TimeZone', 'local'),'HHMM');

%User input -- will be automated once we have Moog 
%prompt = 'Altitude? ';
%altitude = input(prompt);

%prompt = 'Azimuth? ';
%azimuth = input(prompt);

%filename = [saving_dir '' date '_' time '_' num2str(azimuth) '_' num2str(altitude) '.h5'];

filename = [saving_dir '' date '_' time '.h5'];

%set inputs equal to degrees 
analyzer_degrees = [deg_range];

%connect to camera and motor
[src]= Connect_camera_motor_ULTRASIP(2,3,3); 

%move motor and take image
for indx = 1:length(analyzer_degrees)
    
    desired_position = analyzer_degrees(indx);
    current_position = Motor_Position_ULTRASIP(desired_position);

    image(indx,:,:) = Take_image_ULTRASIP(vid,framesPerTrigger,exposure);
   
end

%Save data as .h5 files 
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
%h5writeatt(filename,'/measurement/images/','altitude', altitude);
%h5writeatt(filename,'/measurement/images/','azimuth', azimuth);
h5writeatt(filename,'/measurement/images/','user_notes', usernotes);

end