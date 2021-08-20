% ReadUV_Data.m -- print plots and images from integrating sphere file
%
% Written by Atkin Hyatt 08/19/2021
% Last modified by Atkin Hyatt 08/20/2021

function ReadUV_Data(f, trueDel)
addpath('C:\ULTRASIP\Code\Matlab_Formatting');
addpath('C:\ULTRASIP_Data\July2021\Uncorrected Data');
addpath('C:\ULTRASIP_Data\July2021\Corrected Data');

%% Intermediate Images
image = h5read(f,'/measurement/polarization/radiometric/');

for n = 1 : length(image(:,323,275))/4
    N = 4*n - 3;
    figure(n)
    
    subplot(2,2,1);
    imagesc(image(N));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('0 deg'); caxis([0, 65535])
    
    subplot(2,2,2);
    imagesc(image(N+1));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('45 deg'); caxis([0, 65535])
    
    subplot(2,2,3);
    imagesc(image(N+2));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('90 deg'); caxis([0, 65535])
    
    subplot(2,2,4);
    imagesc(image(N+3));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('135 deg'); caxis([0, 65535])
end

%% Stokes parameters 
stokes = h5read(f,'/measurement/polarization/stokes/');

for m = 1 : length(stokes(:,323,275))/3
    N = 3*m - 2;
    figure(m+n)
    
    subplot(2,2,1);
    imagesc(stokes(N));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S0');
    caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
    
    subplot(2,2,2);
    imagesc(stokes(N+1));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S1');
    caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
    
    subplot(2,2,3);
    imagesc(stokes(N+2));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S2');
    caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
end

%% DoLP and AoLP images
data = h5read(f,'/measurement/polarization/polarizationmetric/');
stdev = h5read(f,'/measurement/polarization/error/');

figure(m+n+1);

subplot(2,1,1)
imagesc(data(1));colorbar;colormap(bone);set(gca,'FontSize',15);axis off;
title('DoLP1 %');caxis([0, 100])

subplot(2,1,2)
imagesc(data(2));colorbar;colormap(bone);set(gca,'FontSize',15);axis off;
title('DoLP2 %');caxis([0, 100])

figure(m+n+2)
subplot(2,1,1)
imagesc(data(3));colorbar;colormap(parula);set(gca,'FontSize',15);
axis off;title('AoLP1');caxis([-90, 90])

subplot(2,1,2)
imagesc(data(4));colorbar;colormap(parula);set(gca,'FontSize',15);
axis off;title('AoLP2');caxis([-90, 90])

%% DoLP and AoLP plots
D1 = mean(mean(data(1,:,:))); D2 = mean(mean(data(2,:,:)));
A1 = mean(mean(data(3,:,:))); A2 = mean(mean(data(4,:,:)));

stdevDOLP = stdev(1,:);
stdevAOLP = stdev(2,:);

figure(m+n+3)
subplot(1,2,1);
errorbar([0 trueDel], [D1 D2], stdevDOLP); title('DoLP vs Delta'); axis([0 trueDel, 0 100]);
xlabel('Delta (deg)'); ylabel('DoLP');

subplot(1,2,2);
errorbar([0 trueDel], [A1 A2], stdevAOLP); title('AoLP vs Delta'); axis([0 trueDel, -90 90]);
xlabel('Delta (deg)'); ylabel('AoLP');
end