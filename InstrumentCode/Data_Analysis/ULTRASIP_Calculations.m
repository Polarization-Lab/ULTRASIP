%% Written by Clarissa M. DeLeon - June 23rd, 2021
%% ULTRASIP_Calculations
%% Calculations performed on flux images: This code performs
%% non-uniformity correction, and calculates stokes parameters, DoLP and AoLP
%close all;  %clear all;

% addpath('C:\ULTRASIP_Data\June2021\FixExposure');
% addpath('C:\ULTRASIP_Data\FPN_Data');
% addpath('C:\ULTRASIP_Data\July2021\Uncorrected Data');
% 
% % Define flux images 
% %******CHANGE PATH******************
% image = h5read(filename,'/measurement/images');

%Separate measurement into the 4 images (0,45,90,135)
range = 1:512;
img0 = squeeze(image(1,range,range));
img45 = squeeze(image(2,range,range));
img90 = squeeze(image(3,range,range));
img135 = squeeze(image(4,range,range));

figure(1);
subplot(2,2,1); imagesc(img0);axis off;title('0 deg');colorbar('FontSize',15);
subplot(2,2,2);imagesc(img45);axis off;title('45 deg');colorbar;
subplot(2,2,3);imagesc(img90);axis off;title('90 deg');colorbar;
subplot(2,2,4);imagesc(img135);axis off;title('135 deg');colorbar;

%Darkfield Correction
%load darkfield
% darkfield = load('darkfield_013secexp.mat').darkfield;
% darkfield = squeeze(darkfield(1,:,:));


%Subtract darkfield from measurement
% img0 = img0 - darkfield;
% img45 = img45 - darkfield;
% img90 = img90 - darkfield;
% img135 = img135 - darkfield;

% Flat Field Correction
flattest = load('FPN_flatfieldSys.mat').flat;

%Linear region 
flattest = flattest(2:26,:,:);

clear m pixelarray B
u = 1:25;
for ii = 1:512
    for jj = 1:512
        for uu = 1:25
            pixelarray(uu) = flattest(uu,ii,jj);
        end
        x = [ones(length(u),1) u'];
        var = x\pixelarray';
        M(ii,jj) = var(2);
        B(ii,jj) = var(1);
    end
end

%Reference slope and intercept
Avg_M = mean(M(:));
Avg_B = mean(B(:));

%Correction for polish marks
gamma0 = fmincon(@(gamma_initial)StdDevCorrected(gamma_initial,img0,Avg_M,Avg_B,M,B),3,[],[],[],[],3,4.5);
img0fix = ImgCorrection(img0,gamma0,M,B,Avg_M,Avg_B);

gamma45 = fmincon(@(gamma_initial)StdDevCorrected(gamma_initial,img45,Avg_M,Avg_B,M,B),3,[],[],[],[],3,4.5);
img45fix = ImgCorrection(img45,gamma45,M,B,Avg_M,Avg_B);

gamma90 = fmincon(@(gamma_initial)StdDevCorrected(gamma_initial,img90,Avg_M,Avg_B,M,B),3,[],[],[],[],3,4.5);
img90fix = ImgCorrection(img90,gamma90,M,B,Avg_M,Avg_B);

gamma135 = fmincon(@(gamma_initial)StdDevCorrected(gamma_initial,img135,Avg_M,Avg_B,M,B),3,[],[],[],[],3,4.5);
img135fix = ImgCorrection(img135,gamma135,M,B,Avg_M,Avg_B);

% For checking if noise remains run these
figure(2);
subplot(2,2,1);imagesc(img0fix);axis off;title('Corrected 0 deg');colorbar;
subplot(2,2,2);imagesc(img45fix);axis off;title('Corrected 45 deg');colorbar;
subplot(2,2,3);imagesc(img90fix);axis off;title('Corrected 90 deg');colorbar;
subplot(2,2,4);imagesc(img135fix);axis off;title('Corrected 135 deg');colorbar;
% Save figures
%*******************CHANGE PATH*******************************
%saveas(figure(1),'C:\ULTRASIP_Data\June2021\FixExposure\original.png')
%saveas(figure(2),'C:\ULTRASIP_Data\June2021\FixExposure\corrected.png');

% S0fix = img0fix./2 + img90fix./2 + img45fix./2 + img135fix./2;
% figure;imagesc(S0fix);axis on;title('S0');colorbar;
 
% NUC Images
img0 = img0fix(50:462,50:462);
img45 = img45fix(50:462,50:462);
img90 = img90fix(50:462,50:462);
img135 = img135fix(50:462,50:462);

img0 = img0fix(190:512,1:275);
img45 = img45fix(190:512,1:275);
img90 = img90fix(190:512,1:275);
img135 = img135fix(190:512,1:275);

% Calculate Stoke's Parameters
S0 = img0./2 + img90./2 + img45./2 + img135./2;
S01 = img0 + img90;
S02 = img45 + img135;

%%%I don't know..Jake did this will double check%%%%%
S0diff = S01 - S02;

S1 = img0 - img90;
S2 = img45 - img135;

s1 = (S1./S0);
s2 = S2./S0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AoLP and DoLP
DoLP = sqrt(S1.^2 + S2.^2)./S0;
AoLP = rad2deg(0.5*atan2(S2,S1));

%AoLPdeg = AoLP*180/pi;
DoLP = DoLP.*100;

