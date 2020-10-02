function numPosition = findpositionESP(device, motor) 
%FINDPOSITION Summary of this function goes here
%James Heath heathjam@email.arizona.edu
%Sept 25 2020
%This function finds the numerical value of the actual position of the
% motor relative to the last zero.  The return is a double, not a string.
%
% The syntax is as follows:
%
% findposition(device, motor)
%
% Device is the declared visa device, and motor is the desired motor or
% axis to be analyzed.


try 
    fopen(device);
end

numPosition = str2double(query(device, strcat(num2str(motor), 'TP')));

fclose(device);