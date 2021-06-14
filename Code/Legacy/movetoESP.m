function position = movetoESP(device, motor, distance)
% FINDPOSITION Summary of this function goes here
% James Heath heathjam@email.arizona.edu
% Sept 25 2020
% Function moves the motor
% directly to a designated position relative to the current 0. This 
% function returns the final position of the motor after displacement as 
% a double, not a string.  A wait is built into this function to avoid
% overloading the controller buffer.
%
% The syntax is as follows:
%
% moveto(device, motor, distance)
%
% Where device is the declared visa device, motor is the desired motor, and
% distance is the distance from the current zero position in millimeters to
% 3 decimal points.  Note that the final position value of this command has
% an error of +/- 0.050.

try
    fopen(device)
end

% The motor and distance variables must be converted to strings and
% concatenated before it's sent to the controller
movecommand = strcat(num2str(motor),'PA',num2str(distance));
wait4stop = strcat(num2str(motor),'WS');
wait4jerk = strcat(num2str(motor),'WT.5');
positionquery= strcat(num2str(motor),'TP?');

% Concantenate and send to the controller
command = strcat(movecommand, ';', wait4stop, ';', wait4jerk, ';', positionquery);
fprintf(device, command);

%find the position
position = str2double(fscanf(device));

fclose(device);