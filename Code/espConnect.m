function esp301 = espConnect(comPort)
% This function will automatically connect the esp301 motion controller
% using the virtual serial port (usually COM3), and configuring it
% appropriately.  To my knowledge, the only two parameters of the ESP301
% which are different from matlab defaults are the baudrate and terminator.
%  These are set below.
%
% James Heath 09/28/2020
%%
esp301 = serial(comPort);
set(esp301, 'baudrate', 921600);
set(esp301, 'terminator', 13);

%%
fopen(esp301)
fprintf(esp301,'VE'); %Sends current wavelength cmd, which is echoed
esp301_Status=fscanf(esp301)
fclose(esp301)