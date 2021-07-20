% Move_motor -- Converts an input angle into the propera hexadecimal string
% to position an ELL14 rotation stage
%
% Written by Atkin Hyatt 06/18/2021
% Last modified by Atkin Hyatt 07/19/2021

function pos = Move_motor(deg,ELL14)

% Constants
pulsPerDeg = 398.222222222;
deviceAddress = '0';

% Open ELL14
if ~isempty(instrfind(ELL14,'Status','closed'))
    %comPort = 'COM1';
    %ELL14 = ELL14Connect(comPort, home);
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

hex = query(ELL14, "0gp");
fclose(ELL14);

% Give real position
%[~, pos] = strtok(q, '0');
%pos = strtok(pos);
%pos = hex2dec(pos) / pulsPerDeg;

pos = TranslateELL14(hex);
fprintf("Actual Position: %0.6f degrees\n", pos);
end
