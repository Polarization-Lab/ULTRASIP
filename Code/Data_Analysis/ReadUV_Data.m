% ReadUV_Data.m -- print plots and images from integrating sphere file
%
% Written by Atkin Hyatt 08/19/2021
% Last modified by Atkin Hyatt 08/20/2021

function ReadUV_Data(f)
addpath('C:\ULTRASIP\Code\Matlab_Formatting');
addpath('C:\ULTRASIP_Data\Data2021\Uncorrected Data');
addpath('C:\ULTRASIP_Data\Data2021\Corrected Data');

%% Intermediate Images
% image = h5read(f,'/measurement/polarization/radiometric/');
% 
% for n = 1 : length(image(:,323,275))/4
%     N = 4*n - 3;
%     figure(n)
%     
%     subplot(2,2,1);
%     imagesc(image(N));set(gca,'FontSize',15);colorbar;
%     colormap(parula);axis off;title('0 deg'); caxis([0, 65535])
%     
%     subplot(2,2,2);
%     imagesc(image(N+1));set(gca,'FontSize',15);colorbar;
%     colormap(parula);axis off;title('45 deg'); caxis([0, 65535])
%     
%     subplot(2,2,3);
%     imagesc(image(N+2));set(gca,'FontSize',15);colorbar;
%     colormap(parula);axis off;title('90 deg'); caxis([0, 65535])
%     
%     subplot(2,2,4);
%     imagesc(image(N+3));set(gca,'FontSize',15);colorbar;
%     colormap(parula);axis off;title('135 deg'); caxis([0, 65535])
% end
% 
% %% Stokes parameters 
% stokes = h5read(f,'/measurement/polarization/stokes/');
% 
% for m = 1 : length(stokes(:,323,275))/3
%     N = 3*m - 2;
%     figure(m+n)
%     
%     subplot(2,2,1);
%     imagesc(stokes(N));set(gca,'FontSize',15);colorbar;
%     colormap(gwp);axis off;title('S0');
%     caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
%     
%     subplot(2,2,2);
%     imagesc(stokes(N+1));set(gca,'FontSize',15);colorbar;
%     colormap(gwp);axis off;title('S1');
%     caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
%     
%     subplot(2,2,3);
%     imagesc(stokes(N+2));set(gca,'FontSize',15);colorbar;
%     colormap(gwp);axis off;title('S2');
%     caxis([-max(max(abs(stokes(N,:,:)))) max(max(abs(stokes(N,:,:))))]);
% end
% 
% %% DoLP and AoLP images
% data = h5read(f,'/measurement/polarization/polarizationmetric/');


% figure(m+n+1);
% 
% subplot(2,1,1)
% imagesc(data(1));colorbar;colormap(bone);set(gca,'FontSize',15);axis off;
% title('DoLP1 %');caxis([0, 100])
% 
% subplot(2,1,2)
% imagesc(data(2));colorbar;colormap(bone);set(gca,'FontSize',15);axis off;
% title('DoLP2 %');caxis([0, 100])
% 
% figure(m+n+2)
% subplot(2,1,1)
% imagesc(data(3));colorbar;colormap(parula);set(gca,'FontSize',15);
% axis off;title('AoLP1');caxis([-90, 90])
% 
% subplot(2,1,2)
% imagesc(data(4));colorbar;colormap(parula);set(gca,'FontSize',15);
% axis off;title('AoLP2');caxis([-90, 90])

%% Extract Data
interImage = h5read(f,'/measurement/polarization/radiometric/');
S = h5read(f,'/measurement/polarization/stokes');
stdevData = h5read(f,'/measurement/polarization/error/');
iter = h5read(f,'/measurement/polarization/datapoints/');
rawData = h5read(f,'/measurement/polarization/polarizationmetric/');

%% Intermediate Images Movie
imVid = VideoWriter('C:\ULTRASIP_Data\Data2021\Data GIFs\IntermediateImages.mp4'); 
imVid.FrameRate = iter / 60;
open(imVid);

for N = 1 : iter
    %% plot images
    figure(1)
    
    subplot(2,2,1);
    imagesc(interImage(N));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('0 deg'); caxis([0, 65535])
    
    subplot(2,2,2);
    imagesc(interImage(N+1));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('45 deg'); caxis([0, 65535])
    
    subplot(2,2,3);
    imagesc(interImage(N+2));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('90 deg'); caxis([0, 65535])
    
    subplot(2,2,4);
    imagesc(interImage(N+3));set(gca,'FontSize',15);colorbar;
    colormap(parula);axis off;title('135 deg'); caxis([0, 65535])
    
    %% movie
    frame = getframe(gcf);
    writeVideo(imVid, frame);
end
close(imVid)

%% Stokes Images Movie
stVid = VideoWriter('C:\ULTRASIP_Data\Data2021\Data GIFs\StokesImages.mp4'); 
stVid.FrameRate = iter / 60;
open(stVid);

for N = 1 : iter
    %% plot stokes
    figure(2)
    subplot(2,2,1);
    imagesc(S(N));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S0');
    caxis([-max(max(abs(S(N,:,:)))) max(max(abs(S(N,:,:))))]);
    
    subplot(2,2,2);
    imagesc(S(N+1));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S1');
    caxis([-max(max(abs(S(N,:,:)))) max(max(abs(S(N,:,:))))]);
    
    subplot(2,2,3);
    imagesc(S(N+2));set(gca,'FontSize',15);colorbar;
    colormap(gwp);axis off;title('S2');
    caxis([-max(max(abs(S(N,:,:)))) max(max(abs(S(N,:,:))))]);
    
    %% movie
    frame = getframe(gcf);
    writeVideo(stVid, frame);
end
close(stVid)

% %% Process DOLP and AOLP
% %DOLP = rawData(1,:,:,:); AOLP = rawData(2,:,:,:);
% D = zeros(1, iter); A = zeros(1,iter);
% 
% for ii = 1 : iter
%     D(ii) = mean(mean(rawData(1,ii,:,:)));
%     A(ii) = mean(mean(rawData(2,ii,:,:)));
% end
% 
% stdevDOLP = stdevData(1,:);
% stdevAOLP = stdevData(2,:);
% 
% del = 0 : 180/(iter-1) : 180;
% 
% figure(1)
% subplot(1,2,1);
% errorbar(del, D, stdevDOLP); title('DoLP vs Delta'); axis([0 180, 0 100]);
% xlabel('Delta (deg)'); ylabel('DoLP');
% 
% subplot(1,2,2);
% errorbar(del, A, stdevAOLP); title('AoLP vs Delta'); axis([0 180, -90 90]);
% xlabel('Delta (deg)'); ylabel('AoLP');

%% S2 vs S1
clear s1 s2 stokes1 stokes2
s1 = S(2 : 3 : 3*iter,:,:) ./ S(1 : 3 : 3*iter,:,:);
s2 = S(3 : 3 : 3*iter,:,:) ./ S(1 : 3 : 3*iter,:,:);

for N = 1 : iter
    stokes1(N) = mean(mean(s1(N,:,:)));
    stokes2(N) = mean(mean(s2(N,:,:)));
end
figure(3)
plot(stokes1,stokes2)
% stokes1 = mean(mean(S((1 : 3 : 3*iter),:,:)));
end