function [dt, azimuth, altitude] = UVsummary_h5(filename)
%summary_h5 this function will return the contents of a stepper HDF5 files
%inputs:
%   filename string of filename and path


%collect and display wavelengths in file 
% info = h5info(filename,'/measurement/images');
% trace_groups = info.Groups;
% groups_names = {trace_groups.Name};
% 
% disp('I dont know what this will do')
% disp(groups_names)

%display info about date & time
date=h5readatt(filename,'/measurement/images','date');
time=h5readatt(filename,'/measurement/images','time');
dt = [num2str(date) '_' num2str(time)];
disp('Date & time')
disp(dt)
disp(' ')

%display info about Angles
azimuth=h5readatt(filename,'/measurement/images','azimuth');
disp('Azimuthal angle')
disp(azimuth)

altitude=h5readatt(filename,'/measurement/images','altitude');
disp('Altitude angle')
disp(altitude)

%Measurement time
Meas_Time=h5readatt(filename,'/measurement/images','meas_time');
disp('Measurement total time')
disp(Meas_Time)

%display user notes
notes=h5readatt(filename,'/measurement/images','user_notes');
disp('User notes :')
disp(notes)
end

