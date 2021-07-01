% Converts an input angle into a hexadecimal string to position an ELL14
% rotation stage as well as outputs the true position of the stage

% Constants
% pulsPerDeg - number of pulses per degree, 398 + 2/9 for ELL14
% comPort - COM port
% deviceAddress - device address (default 0)
% ELL14 - ELL14 object

% Variables
% deg - user inputted degrees
% pulses - number of pulses needed to move ELL14 by deg
% angleCommand - fourth byte needed to position the stage

% Written by Atkin Hyatt 06/18/2021

function y = Move_motor(deg,ELL14)

% Constants
pulsPerDeg = 398.222222222;
deviceAddress = '0';

% Open ELL14
if ~isempty(instrfind(ELL14,'Status','closed'))
    comPort = 'COM1';
    ELL14 = ELL14Connect(comPort);
    fopen(ELL14);
%    disp('ELL14 now open')
%else
%    disp('ELL14 already open')
end

% Convert to hexadecimal
pulses = round(pulsPerDeg * deg);
angleCommand = dec2hex(pulses, 8);

% Build command and export
y = deviceAddress + "ma" + angleCommand;    % move absolute
fprintf(ELL14,"%s",y);

q = query(ELL14, "0gp");
fclose(ELL14);

% Give real position
[~, pos] = strtok(q, '0');
pos = strtok(pos);
pos = hex2dec(pos) / pulsPerDeg;
fprintf("Actual Position: %f degrees", pos);
end
