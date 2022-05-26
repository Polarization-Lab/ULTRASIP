% EncodeELL14.m -- Converts input angular position to hex code for ELL14 to
% interpret
%
% Written by Atkin Hyatt
% 

function hex = EncodeELL14(angle)

conv = round(angle * 398.2222222);
hex = dec2hex(conv, 8);

end