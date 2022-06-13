function [] = NI_shutter_UV(position)
%NI_SHUTTER this function will open and close the external shutter
%   Note this only works in R2020 and beyond
%   Until Hamamatsu driver works in R2020, this will need to be executed
%   externally to the general imaging chain
%   position - bool, if true shutter opens, close if F

%create a DataAquisition and Add Analog input channels 
ai = daq('ni') ;
addoutput(ai,'Dev2','port1/line0','Digital');

% Write values to output channels (turns on, waits for a return, turns off)
if position
    output = 0;
else 
    output = 1;
end

write(ai,output)
pause(5)

end