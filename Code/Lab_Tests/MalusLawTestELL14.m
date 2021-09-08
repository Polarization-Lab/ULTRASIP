% Malus Law Test
% Written by Caroline Humphreys 07/26/2021
% Move motor in small increments to find the degree at which minimum
% intensity occurs (extinction ratio test)

% Initialize
addpath('C:\ULTRASIP\Code\Taking_Data!!!');
initializeUV

% Configure camera
triggerconfig(vid, 'manual');
disp('Taking data')

%Start camera comms
start(vid)

%Take images
% tic
for ii = 1:720
%     tic   
    fprintf('Taking data %0.3f Degrees\n',ii/2);  

    fprintf(ELL14,sprintf('%0.3f',ii/2));
    Move_motor(ii/2,ELL14);
    %different loop values ^    
    fopen(ELL14);
    TranslateELL14(query(ELL14,sprintf("0gp"))) %ask position

    image(ii,:,:) = UV_data(vid,framesPerTrigger); %take picture
    Intensity(ii) = max(image(ii,:));
    %pause(2);
    disp('image taken')
    
%     eachtime(ii) = toc;
end
% finaltime = toc;

%Close instruments
fclose(ELL14);
stop(vid)
disp('Done')

% plot intensity graph and find max value
plot(Intensity)
title('Intensity');
xlabel('Degrees');
i0 = max(Intensity)/2;
[maxYValue, indexAtMaxY] = max(Intensity);
xValueAtMaxYValue = ii(indexAtMaxY(1));