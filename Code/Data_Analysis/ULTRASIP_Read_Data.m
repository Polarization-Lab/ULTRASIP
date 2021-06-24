%% Written by Clarissa M. DeLeon - June 23rd, 2021
%% ULTRASIP_Read_Data
%% Code to look at ULTRASIP Analyzed Data saved in h5 file
%% NOTE: Run ULTRASIP_Auto_Analysis First 

clear all; close all;

%*******CHANGE FILENAME************%
% Define h5 file 
filename = '2021-06-23_1749_0.1.h5'; 
%Display h5 file
h5disp(filename);

%Define Chunks to display (will automate this later)--CD :S
rows = 1:323;
columns1 = 1:275;
columns2 = 276:550;
columns3 = 551:825;
columns4 = 826:1100;

%%Look at flux images
image = h5read(filename,'/measurement/polarization/radiometric/');

%Separate images 
img0 = squeeze(image(rows,columns1));
img45 = squeeze(image(rows,columns2));
img90 = squeeze(image(rows,columns3));
img135 = squeeze(image(rows,columns4));

%Display images
figure;imagesc(img0);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;title('0 deg');
figure;imagesc(img90);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;title('90 deg');
figure;imagesc(img45);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;title('45 deg');
figure;imagesc(img135);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;title('135 deg');

%%Look at stokes parameters 
stokes = h5read(filename,'/measurement/polarization/stokes/');

%Separate Parameters 
S0 = squeeze(stokes(rows,columns1));
s1 = squeeze(stokes(rows,columns2));
s2 = squeeze(stokes(rows,columns3));

%Display Stokes
figure;imagesc(S0);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('S0');
figure;imagesc(s1);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S1');caxis([-max(abs(s1(:))) max(abs(s1(:)))]);
figure;imagesc(s2);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S2');caxis([-max(abs(s2(:))) max(abs(s2(:)))]);

%%Look at AoLP and DoLP
linearpol = h5read(filename,'/measurement/polarization/polarizationmetric/');

%Separate Parameters 
DoLP = squeeze(linearpol(rows,columns1));
AoLP = squeeze(linearpol(rows,columns2));

%Display AoLP and DoLP --need to fix up how these are presented--CD :S
figure;imagesc(DoLP);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('DoLP %');
figure;imagesc(AoLP);set(gca,'FontSize',15);axis off;title('AoLP');colorbar;

