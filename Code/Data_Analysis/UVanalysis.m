%%UV Analysis Code

%Load measurement
%image = load('1653.mat').image;
image = load('clouds_1451_n401749_2034923.mat').image;

%clouds_0940_n592141_102701.mat <meas2
%clouds_1025_n264741_180448.mat <- Good one! meas1
%clouds_1521_n472353_1934711.mat <- sun image meas3

%Separate measurement into 4 orthogonal images (0,45,90,135)
range = 1:512;
img0 = squeeze(image(1,range,range));
img45 = squeeze(image(2,range,range));
img90 = squeeze(image(3,range,range));
img135 = squeeze(image(4,range,range));

% %load darkfield
% darkfield = load('darkfield_013secexp.mat').darkfield;
% darkfield = squeeze(darkfield(1,:,:));


%Subtract darkfield from measurement
% img0 = img0 - darkfield;
% img45 = img45 - darkfield;
% img90 = img90 - darkfield;
% img135 = img135 - darkfield;

%Don't use caxis for these measurements
% Noise Correction
flattest = load('FPN_flatfieldSys.mat').flat;

flattest = flattest(2:26,:,:);

clear m pixelarray B
u = 1:24;
for ii = 1:512
    for jj = 1:512
        for uu = 1:24
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
%
%weighting variable

%
%Correction for polish marks
close all;
gamma0 = 3.5;
gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,img0,Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
img0fix = ImgCorrection(img0,gamma0,M,B,Avg_M,Avg_B);

gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,img45,Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
img45fix = ImgCorrection(img45,gamma0,M,B,Avg_M,Avg_B);

gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,img90,Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
img90fix = ImgCorrection(img90,gamma0,M,B,Avg_M,Avg_B);

gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,img135,Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
img135fix = ImgCorrection(img135,gamma0,M,B,Avg_M,Avg_B);

% For checking if noise remains
% figure;imagesc(img0fix);axis off;colorbar;%title('Corrected 0 deg');colorbar;
% figure;imagesc(img45fix);axis off;%title('Corrected 45 deg');colorbar;
% figure;imagesc(img90fix);axis off;%title('Corrected 90 deg');colorbar;
% figure;imagesc(img135fix);axis off;%title('Corrected 135 deg');colorbar;
% 
% S0fix = img0fix./2 + img90fix./2 + img45fix./2 + img135fix./2;
% 
% figure;imagesc(S0fix);axis on;title('S0');colorbar;

%% AOI
img0 = img0fix(50:462,50:462);
img45 = img45fix(50:462,50:462);
img90 = img90fix(50:462,50:462);
img135 = img135fix(50:462,50:462);
%%
img0 = img0fix(190:512,1:275);
img45 = img45fix(190:512,1:275);
img90 = img90fix(190:512,1:275);
img135 = img135fix(190:512,1:275);

%%
%Use caxis for these measurements
S0 = img0./2 + img90./2 + img45./2 + img135./2;
S01 = img0 + img90;
S02 = img45 + img135;

% img135 = S01 - img45;


S0diff = S01 - S02;

S1 = img0 - img90;
S2 = img45 - img135;

%pics for below
s1 = (S1./S0);
s2 = S2./S0;

DoLP = sqrt(S1.^2 + S2.^2)./S0;
% DoLPTest = sqrt(S1.^2 + S2Test.^2)./test;
AoLP = 0.5*atan2(S2,S1);

% AoLPTest = 0.5*atan2(S2Test,S1);
AoLPdeg = AoLP*180/pi;

DoLP = DoLP.*100;
% 
% % close all;
figure;imagesc(img0);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;%title('0 deg');%caxis([min(img90(:)) max(img0(:))]);%
figure;imagesc(img90);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;%title('90 deg');%caxis([min(img90(:)) max(img0(:))]);%title('90 deg');caxis([min(img90(:)) max(img0(:))]);
figure;imagesc(img45);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;%title('45 deg');%caxis([min(img90(:)) max(img0(:))]);%title('45 deg');caxis([min(img90(:)) max(img0(:))]);
figure;imagesc(img135);set(gca,'FontSize',15);colorbar;colormap(parula);axis off;%title('135 deg');%caxis([min(img90(:)) max(img0(:))]);%title('135 deg');caxis([min(img90(:)) max(img0(:))]);
figure;imagesc(S0);set(gca,'FontSize',15);colorbar;colormap(bone);axis off;%title('S0');
figure;imagesc(s1);set(gca,'FontSize',15);colorbar;colormap(gwp);axis off;caxis([-max(abs(s1(:))) max(abs(s1(:)))]);title('S1');
figure;imagesc(s2);set(gca,'FontSize',15);colorbar;colormap(gwp);axis off;caxis([-max(abs(s2(:))) max(abs(s2(:)))]);title('S2');
figure;imagesc(s1);set(gca,'FontSize',15);colorbar;colormap(bone);axis off;
figure;imagesc(s2);set(gca,'FontSize',15);colorbar;colormap(bone);axis off;
figure;imagesc(DoLP);set(gca,'FontSize',15);colorbar;colormap(bone);axis off;%title('DoLP'); caxis([0 max(abs(DoLP(:)))]);
figure;imagesc(AoLPdeg);axis equal;axis off; phasemap(3,'deg'); phasebar; title('AoLP [deg]');
figure;imagesc(AoLPdeg);set(gca,'FontSize',15);axis off;colorbar;caxis([-14 -7]); %title('Scaled AoLP');
 disp('done')
% %%
% %img0 = imrotate(img0,90);
% %img45 = imrotate(img45,90);
% %img90 = imrotate(img90,90);
% %img135 = imrotate(img135,90);
% disp('figures')
% close all;
% figure;imagesc(img0);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('0 deg');
% figure;imagesc(img90);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('90 deg');
% figure;imagesc(img45);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('45 deg');
% figure;imagesc(img135);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('135 deg');
% figure;imagesc(S0);colorbar;colormap(parula);set(gca,'FontSize',15);axis off;title('S0');
% figure;imagesc(s1);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('S1 scaled');%caxis([-max(abs(s1(:))) max(abs(s1(:)))]);
% figure;imagesc(s2);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('S2 scaled');%caxis([-max(abs(s2(:))) max(abs(s2(:)))]);
% figure;imagesc(s1);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S1');caxis([-max(abs(s1(:))) max(abs(s1(:)))]);
% figure;imagesc(s2);colorbar;colormap(gwp);set(gca,'FontSize',15);axis off;title('S2');caxis([-max(abs(s2(:))) max(abs(s2(:)))]);
% figure;imagesc(DoLP);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('DoLP scaled');
% figure;imagesc(DoLP);colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('DoLP');caxis([0 max(abs(DoLP(:)))]);
% figure;imagesc(AoLPdeg);axis equal;set(gca,'FontSize',15);axis off; phasemap; phasebar; title('AoLP [deg]');
% figure;imagesc(AoLPdeg);set(gca,'FontSize',15);axis off;title('Scaled AoLP');colorbar;
