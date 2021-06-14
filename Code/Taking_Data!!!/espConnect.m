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
%Ask for position
fopen(esp301);
global CURRENT_POS;
response = query(esp301,'1TP?')


if ~isempty(response) && length(response) == 3
    CURRENT_POS = response;
else
    success = false;
    return;
end

%If not homed, home
if CURRENT_POS ~= 0
    fprintf(esp301,'1OR');
end

%Set speed and acceleration/decceleration for axis 1
global CURRENT_SPEED;
if isempty(CURRENT_SPEED)
    CURRENT_SPEED = 90;
end

global CURRENT_ACCEL;
if isempty(CURRENT_ACCEL)
    CURRENT_ACCEL = 50;
end

global CURRENT_DECEL;
if isempty(CURRENT_DECEL)
    CURRENT_DECEL = 50;
end

fprintf(esp301,sprintf('%dVA%0.5f', 1, CURRENT_SPEED)); % Set axis 1 velocity
fprintf(esp301,sprintf('%dAC%0.5f', 1, CURRENT_ACCEL)); % Set axis 1 accel
fprintf(esp301,sprintf('%dAG%0.5f', 1, CURRENT_DECEL)); % Set axis 1 decel

fclose(esp301)

%% NEED TO ADD HOMING COMMANDS