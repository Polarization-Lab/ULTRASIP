%% Script to move motor 
%% Calls Move_motor.m 

%Connect to ELL14 by calling ELL14Connect.m
% Connect to piezo Motor
comPort = 'COM1'; %Whichever port the ESP301 is plugged in
ELL14 = ELL14Connect(comPort);

disp('connected')

%Check if ELL14 is open
if ~isempty(instrfind(ELL14,'Status','close'))
    fopen(ELL14);
    disp('ELL14 now open')
else
    disp('ELL14 already open')
end

%Move motor by calling Move_motor.m
deg = 10; %put in desired degree
y = Move_motor(deg);
fprintf(ELL14,'%s',y);

%Close instruments
fclose(ELL14);
disp('done')