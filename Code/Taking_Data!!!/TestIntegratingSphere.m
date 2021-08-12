% TestIntegratingSphere.m -- Take integrating sphere measurements with
% offset of 0 as well as offset of some arbitrary delta.
%
% Written by Atkin Hyatt 08/06/2021
% Last modified by Atkin Hyatt 08/10/2021

% Initialize
%initializeUV
expo = input('Exposure time in seconds? ');
src.ExposureTime = expo;

if expo >= 5
   vid.Timeout = 2 * expo;
end

delta = input('Delta value in degrees? ');

% Configure camera
triggerconfig(vid, 'manual');

%% Take data
% Start camera comms
start(vid);

% initialize variables
deg = [0, 45, 90, 135];
L = length(deg);
image = zeros(2*L,512,512);

% First data point
for N = 1 : L
    Move_motor(deg(N),ELL14);

    pause(1)
    
    image(N,:,:) = UV_data(vid,framesPerTrigger) - dark;
    disp('image taken')
end

% "Tilt" instrument to delta
fopen(ELL14);
h = dec2hex(round(398.222222222222 * delta + hex2dec(home)), 8);
fprintf(ELL14, "%s", "0so" + h);
fclose(ELL14);

% Second data point
for N = 1 : L
    Move_motor(deg(N),ELL14);
    
    pause(1)
    
    image(N + L,:,:) = UV_data(vid,framesPerTrigger) - dark;
    disp('image taken')
end

%Close instruments
stop(vid)
fopen(ELL14);
fprintf(ELL14, "%s", "0so" + home);
fclose(ELL14);

%% Save data
% Define saving  directory 
saving_dir = 'C:\ULTRASIP_Data\July2021\Uncorrected Data\';

% Create data directory 
date = datestr(now,'yyyy-mm-dd'); % get date
time = datestr(datetime('now', 'TimeZone', 'local'),'HHMM'); % get time

% Convert decimals to dashes
ex = char(string(expo));
for N = 1 : length(ex)
   if ex(N) == '.'
       ex(N) = '-';
       break
   end
end

del = char(string(delta));
for N = 1 : length(del)
   if del(N) == '.'
       del(N) = '-';
       break
   end
end

global filename
filename = [saving_dir '' date '_' time '_' ex '_' del '.h5'];

% Create h5
h5create(filename,'/measurement/images',size(image));
h5write(filename,'/measurement/images',image);

% Write attibutes to directory
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','delta', delta);
h5writeatt(filename,'/measurement/images/','exposure', expo);

%% Data processing
addpath('C:\ULTRASIP\Code\Data_Analysis');
ULTRASIP_IntegratingSphere_Analysis;

for N = 1 : length(correctImage(:,323,275))/4
    stdevDOLP(N) = std(reshape(DoLP, 1, 323*275));
    stdevAOLP(N) = std(reshape(AoLP, 1, 323*275));
end

%% Plot data
% Show intermediate images

for n = 1 : length(correctImage(:,323,275))/4
    N = 4*n - 3;
    figure(n)
    subplot(2,2,1);
    imagesc(correctImage(N));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('0 deg'); caxis([0, 65535])
    
    subplot(2,2,2);
    imagesc(correctImage(N+1));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('90 deg'); caxis([0, 65535])
    
    subplot(2,2,3);
    imagesc(correctImage(N+2));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('45 deg'); caxis([0, 65535])
    
    subplot(2,2,4);
    imagesc(correctImage(N+3));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('135 deg'); caxis([0, 65535])
end

figure(3)
subplot(1,2,1);
errorbar([0 delta], [D1 D2], stdevDOLP); title('DoLP vs Delta'); axis([0 delta, 0 100]);
xlabel('Delta (deg)'); ylabel('DoLP');

subplot(1,2,2);
errorbar([0 delta], [A1 A2], stdevAOLP); title('AoLP vs Delta');
xlabel('Delta (deg)'); ylabel('AoLP');