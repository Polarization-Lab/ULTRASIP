
%Automatic Analysis
clear all; close all;

%% For Data before May 5th 2021
% Define data directory 
myFolder = 'C:\ULTRASIP\Data\March2021\Rooftop\March_29';
      filePattern = fullfile(myFolder, '*.mat');
      matFiles = dir(filePattern);
      cd(myFolder)
      maximum = 0;
 
      
  for i = 1:length(matFiles)
     cd(myFolder)
     %pause(2);
     %load .mat file
     baseFileName = fullfile(myFolder, matFiles(i).name);
     load(baseFileName);
     %make folder for processed data
     j = matFiles(i).name;
     name=[num2str(j) '_Processed'];%new folder name
     if ~exist(name, 'dir')
     mkdir(name);%create new folder
    
     %set folder to save data
     savefolder = fullfile(myFolder, name);
     addpath(savefolder)
     cd(savefolder)
     
     disp('starting analysis')
    UVanalysis; %Run analysis 

disp('done')
  
  
disp('figures')
close all;
figure(1);imagesc(img0);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('0 deg');
figure(2);imagesc(img90);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('90 deg');
figure(3);imagesc(img45);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('45 deg');
figure(4);imagesc(img135);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('135 deg');
figure(5);imagesc(S0);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('S0');
figure(6);imagesc(s1);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('S1 scaled');%caxis([-max(abs(s1(:))) max(abs(s1(:)))]);
figure(7);imagesc(s2);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('S2 scaled');%caxis([-max(abs(s2(:))) max(abs(s2(:)))]);
figure(8);imagesc(s1);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S1');caxis([-max(abs(s1(:))) max(abs(s1(:)))]);
figure(9);imagesc(s2);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S2');caxis([-max(abs(s2(:))) max(abs(s2(:)))]);
figure(10);imagesc(DoLP);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('DoLP %');
figure(11);imagesc(DoLP);colorbar;colormap(hot);set(gca,'FontSize',15);axis off;title('DoLP');caxis([0 40]);%max(abs(DoLP(:)))]);
 figure(12);imagesc(AoLPdeg);axis equal;set(gca,'FontSize',15);axis off; phasemap; phasebar; title('AoLP [deg]');
 figure(13);imagesc(AoLPdeg);set(gca,'FontSize',15);axis off;title('AoLP');colorbar;

%test variables
% img0 = zeros(512,512);
% img45 = img0;
% img90 = img0;
% img135 = img0;
% S0 = img0;
% s1 = img0;
% s2 = img0;
% DoLP = img0;
% AoLP = img0;



saveas(figure(1),'0deg.jpg');
saveas(figure(2),'90deg.jpg');
saveas(figure(3),'45deg.jpg');
saveas(figure(4),'135deg.jpg');
saveas(figure(5),'S0.jpg');
saveas(figure(6),'S1scaled.jpg');
saveas(figure(7),'S2scaled.jpg');
saveas(figure(8),'S1.jpg');
saveas(figure(9),'S2.jpg');
saveas(figure(10),'DoLP %.jpg');
saveas(figure(11),'DoLP.jpg');
saveas(figure(12),'AoLP.jpg');
saveas(figure(13),'AoLP [deg].jpg');

% Video Code 
% videoArray(i,:,:) = DoLP;
% else
%          disp('next')
  
   end
   end
       close all;
       
 %% For data after May 5th, 2021
 filename = '2021-05-13_0854_90_49.h5'; %User input (h5 file you want to look at)
 image = h5read(filename,'/measurement/images');
 disp('starting analysis')
 UVanalysis; %Run analysis 
 %create new file name for polarization data
h5create(filename,'/measurement/polarization/radiometric',size(img0));
h5create(filename,'/measurement/polarization/stokes',size(S0));
h5create(filename,'/measurement/polarization/polarizationmetric',size(DoLP));

h5write(filename,'/measurement/polarization/radiometric/',img0);
h5write(filename,'/measurement/polarization/radiometric/',img45);
h5write(filename,'/measurement/polarization/radiometric/',img90);
h5write(filename,'/measurement/polarization/radiometric/',img135);

h5write(filename,'/measurement/polarization/stokes/',S0);
h5write(filename,'/measurement/polarization/stokes/',s1);
h5write(filename,'/measurement/polarization/stokes/',s2);

h5write(filename,'/measurement/polarization/polarizationmetric/',DoLP);
h5write(filename,'/measurement/polarization/polarizationmetric/',AoLP);

%write attibutes to directory
h5writeatt(filename,'/measurement/images/','meas_time', finaltime);
h5writeatt(filename,'/measurement/images/','date', date);
h5writeatt(filename,'/measurement/images/','time', time);
h5writeatt(filename,'/measurement/images/','altitude', altitude);
h5writeatt(filename,'/measurement/images/','azimuth', azimuth);
h5writeatt(filename,'/measurement/images/','user_notes', usernotes);
  