function ELL14 = ELL14Connect(comPort)
% This function will automatically connect the esp301 motion controller
% using the virtual serial port (usually COM3), and configuring it
% appropriately.  To my knowledge, the only two parameters of the ESP301
% which are different from matlab defaults are the baudrate and terminator.
%  These are set below.
%
% James Heath 09/28/2020
%%
ELL14 = serial(comPort);
set(ELL14, 'baudrate', 9600,'databits',8,'stopbits',1);

%%
%Ask for position
fopen(ELL14);

%set home position and home
fprintf(ELL14,'0so00000000');
query(ELL14,'0ho1');

%set jog to 45 degrees
%fprintf(ELL14,'0sj00004600');
fclose(ELL14);



%% NEED TO ADD HOMING COMMANDS