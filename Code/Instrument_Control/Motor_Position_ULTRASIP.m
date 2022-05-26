%Move motor to desired position 
%Written by Clarissa DeLeon 1.14.2022 

% Use this function to set ELL14K motor position 
%Motor details: Thorlabs Rotation Mount and interface board
    %MODEL: ELL14K

%Funtion in/outputs:
    %desired position is the input to this function in degrees! 
    %current position is where the motor moves to at the end of the sequence -
    %should match inputted postion within moving accuracy 

%MOTOR MUST BE CONNECTED-- run Connect_camera_motor_ULTRASIP

function [current_position]= Motor_Position_ULTRASIP(desired_position)

% Device specs: pulseperdeg found in datasheet 
pulsPerDeg = 398.222222222;
deviceAddress = '0';

%Convert to pulses
pulses = round(pulsPerDeg * desired_position);

% Build command and send to motor
command = deviceAddress + "ma" + dec2hex(pulses, 8);    % move absolute
fprintf(ELL14,"%s",command);

%get current positon in degrees 
pos_in_hex = query(ELL14, "0gp");

current_position = TranslateELL14(hex);

end 