% IntegratingSphere_Correction -- Correct polishing marks on Hamamatsu
% camera, output these images as well as DoLP and AoLP information
%
% Written by Atkin Hyatt 08/09/2021
% Last modified by Atkin Hyatt 08/10/2021

function [correctImage, DoLP, AoLP] = IntegratingSphere_Correction(f, n)
addpath('C:\ULTRASIP_Data\FPN_Data');
addpath('C:\ULTRASIP_Data\July2021\Uncorrected Data');

image = h5read(f,'/measurement/images');

% Separate measurement into the 4 images (0,45,90,135)
range = 1:512;
N = 4*n - 3;

img0 = squeeze(image(N,range,range));
img45 = squeeze(image(N+1,range,range));
img90 = squeeze(image(N+2,range,range));
img135 = squeeze(image(N+3,range,range));

global flattest
flattest = load('FPN_flatfieldSys.mat').flat;

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

img0 = img0fix(50:462,50:462);
img45 = img45fix(50:462,50:462);
img90 = img90fix(50:462,50:462);
img135 = img135fix(50:462,50:462);

img0 = img0fix(190:512,1:275);
img45 = img45fix(190:512,1:275);
img90 = img90fix(190:512,1:275);
img135 = img135fix(190:512,1:275);

% Stokes parameters
S0 = img0./2 + img90./2 + img45./2 + img135./2;
S1 = img0 - img90;
S2 = img45 - img135;

DoLP = sqrt(S1.^2 + S2.^2)./S0 * 100;
AoLP = rad2deg(0.5*atan2(S2,S1));

% Output fixed images
correctImage(1,:,:) = img0;
correctImage(2,:,:) = img45;
correctImage(3,:,:) = img90;
correctImage(4,:,:) = img135;
end