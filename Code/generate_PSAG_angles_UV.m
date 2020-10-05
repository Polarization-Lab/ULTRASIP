function [PSA, PSG] = generate_PSAG_angles(num)
%GENERATE_PSAG_ANGLES generate arrays of PSG/PSA angles 

rate = 4.91;

size = 360/num ;%PSG angle steps

PSG = 0:size:360-size;

%build PSA array
i = 2;
PSA = zeros(num,1);
while i < num + 1
    PSA(i) = PSA(i-1)+ rate * size;
    if PSA(i) > 360 %keep values etween 0 and 360
        PSA(i) = PSA(i) - 360;
    end
    i = i + 1;
end  

PSA = -PSA;
end

