% AlignUV.m -- Measures the AoLP of the glan-thompson polarizer for proper
% alignment of the instrument
%
% Written by Atkin Hyatt 08/12/2021
% Last modified by Atkin Hyatt 08/20/2021

addpath('C:\ULTRASIP_Data\FPN_Data');
addpath('C:\ULTRASIP\Code\Data_Analysis');

stop(vid)

N = input('How many points to average? ');
A = zeros(1,N); image = zeros(2*L,512,512);

deg = [0, 45, 90, 135]; L = length(deg);

triggerconfig(vid, 'manual');
src.ExposureTime = input('Exposure time in seconds: ');
start(vid);

for Z = 1 : N
    fprintf("\n")
    for N = 1 : L
        Move_motor(deg(N),ELL14);

        pause(2)
    
        image(N,:,:) = UV_data(vid,framesPerTrigger) - dark;
        disp('image taken')
    end
    
    [~, ~, AoLP] = IntegratingSphere_Correction(image, 1, "pic");

    A(Z) = mean(mean(AoLP));
end
hex = dec2hex(round(398.222222222 * mean(A)),8);

newHome = dec2hex(round(398.222222 * (TranslateELL14(home) + TranslateELL14(hex))),8);

fprintf("\nAverage offset of %f degrees\nSet home to %s\n", mean(A), newHome);
stop(vid)
toc