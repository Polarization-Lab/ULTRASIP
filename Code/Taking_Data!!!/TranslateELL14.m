% TranslateELL14 -- interpret position outputs from the ELL14 and convert
% to decimal degrees also taking two's compliment into account
%
% Written by Atkin Hyatt 07/08/2021
% Last modified by Atkin Hyatt 07/19/2021

function decPos = TranslateELL14(hex)
    pulsPerDeg = 398.22222222222;
    if hex(4) == '0'            % if position is positive
        % extract position data
        [~, pos] = strtok(hex, '0');
        pos = strtok(pos);
        
        % convert to degrees in decimal
        decPos = hex2dec(pos) / pulsPerDeg;
    elseif hex(4) == 'F'        % if postion is negative
        % extract position data (two's compliment)
        [~, pos] = strtok(hex, 'F');
        pos = strtok(pos);
        
        % convert two's compliment
        for N = 1 : 8
            pos(N) = 118 - pos(N);
            if pos(N) >= 58 && pos(N) <= 64
               pos(N) = pos(N) - 7; 
            end
        end
        
        % add 1 to hex number
        pos(8) = pos(8) + 1;
        for N = 8 : -1 : 1
            if pos(N) == 'G'
                pos(N) = '0';
                pos(N - 1) = pos(N - 1) + 1;
            end
            if pos(N) == ':'
                pos(N) = 'A';
            end
        end
        
        % convert to degree in decimal, taking negative into account
        decDeg = hex2dec(pos);
        decPos = -1 * decDeg / pulsPerDeg;
    end
end