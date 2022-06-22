%% Home polarizers
if ~isempty(instrfind(esp301,'Status','close'))
    fopen(esp301);
end

%Send home command
fprintf(esp301,'1OR;1WS');

%close connection
fclose(esp301);
 
%Check if ELL14 is open
if ~isempty(instrfind(ELL14,'Status','close'))
    fopen(ELL14);
end

query(ELL14,'0ho1') %home piezo motor  
fclose(ELL14); %close ELL14


%% Run measurement

 %initial GT position
PolState =  [0 45 90 135];
triggerconfig(vid, 'manual'); %Camera trigger control for faster image acquisition

%Start camera comms
start(vid)

for ii = 1:length(PolState)
    
    disp('Begin measurement')
    
    if ~isempty(instrfind(esp301,'Status','close'))
    fopen(esp301);
    end    
    
    Degree = 0; %GT state
    
    %run through GT states
    for jj = 1:36
        
        image(jj,:,:) = UV_data(vid,framesPerTrigger); %take picture
        average(jj) = mean(squeeze(image(jj,50:462,50:462)),'all'); %grab ROI area average
        
        
        Degree = Degree + 10
        pause(0.25)
        fprintf(esp301,sprintf('1PA%0.3f',Degree)); %rotate polarizer for next measurement
        query(esp301,sprintf('1WS;1TP?')); %ask position
    end
    
    save(['State2_' num2str(PolState(ii)) '.mat'],'image');
    
    %Send home command
    fprintf(esp301,'1OR;1WS');
    
    pause(10) %pause 30 secs to ensure motor has homed
    
    %close connection
    fclose(esp301);
    
    %Check if ELL14 is open
    if ~isempty(instrfind(ELL14,'Status','close'))
        fopen(ELL14);
    end
    fprintf(ELL14,'0fw'); %jog wire-grid 45 degrees
    fclose(ELL14); %close ELL14
    
end
stop(vid)

%% Load Data

img0 = load('State_0.mat').img0;
img45 = load('State_45.mat').img45;
img90 = load('State_90.mat').img90;
img135 = load('State_135.mat').img135;
darkfield = load('darkfield_6.mat').darkfield;
%% Second GT

img0 = load('State2_0.mat').image;
img45 = load('State2_45.mat').image;
img90 = load('State2_90.mat').image;
img135 = load('State2_135.mat').image;
%%
for ii = 1:36
    img0(ii,:,:) = squeeze(img0(ii,:,:)) - squeeze(darkfield(1,:,:));
    img45(ii,:,:) = squeeze(img45(ii,:,:)) - squeeze(darkfield(1,:,:));
    img90(ii,:,:) = squeeze(img90(ii,:,:)) - squeeze(darkfield(1,:,:));
    img135(ii,:,:) = squeeze(img135(ii,:,:)) - squeeze(darkfield(1,:,:));
end
%% Analysis

%clear prior variables
clear average0 average45 average90 average135 stdev0 stdev45 stdev90 stdev135 
 
%Grab average data for uncorrected image
for jj=1:36
    average0(jj) = mean(squeeze(img0(jj,50:462,50:462)),'all'); %grab ROI area average
    stdev0(jj) = std2(squeeze(img0(jj,50:462,50:462)));
  
    average45(jj) = mean(squeeze(img45(jj,50:462,50:462)),'all');
    stdev45(jj) = std2(squeeze(img45(jj,50:462,50:462)));
    
    average90(jj) = mean(squeeze(img90(jj,50:462,50:462)),'all');
    stdev90(jj) = std2(squeeze(img90(jj,50:462,50:462)));
    
    average135(jj) = mean(squeeze(img135(jj,50:462,50:462)),'all');
    stdev135(jj) = std2(squeeze(img135(jj,50:462,50:462)));
    
end

%% Image correction

%Don't use caxis for these measurements
% Noise Correction
flattest = load('FPN_flatfieldSys.mat').flat;

flattest = flattest(2:26,:,:);
for kk = 1:36
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

    gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,squeeze(img0(kk,:,:)),Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
    img0fix(kk,:,:) = ImgCorrection(squeeze(img0(kk,:,:)),gamma0,M,B,Avg_M,Avg_B);

    gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,squeeze(img45(kk,:,:)),Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
    img45fix(kk,:,:) = ImgCorrection(squeeze(img45(kk,:,:)),gamma0,M,B,Avg_M,Avg_B);

    gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,squeeze(img90(kk,:,:)),Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
    img90fix(kk,:,:) = ImgCorrection(squeeze(img90(kk,:,:)),gamma0,M,B,Avg_M,Avg_B);

    gamma0 = fmincon(@(gamma0)StdDevCorrected(gamma0,squeeze(img135(kk,:,:)),Avg_M,Avg_B,M,B),2.55,[],[],[],[],1,7);
    img135fix(kk,:,:) = ImgCorrection(squeeze(img135(kk,:,:)),gamma0,M,B,Avg_M,Avg_B);

end

%% Corrected values

for jj=1:36
    average0fix(jj) = mean(squeeze(img0fix(jj,50:462,50:462)),'all'); %grab ROI area average
    stdev0fix(jj) = std2(squeeze(img0fix(jj,50:462,50:462)));
    
    average45fix(jj) = mean(squeeze(img45fix(jj,50:462,50:462)),'all');
    stdev45fix(jj) = std2(squeeze(img45fix(jj,50:462,50:462)));
    
    average90fix(jj) = mean(squeeze(img90fix(jj,50:462,50:462)),'all');
    stdev90fix(jj) = std2(squeeze(img90fix(jj,50:462,50:462)));
    
    average135fix(jj) = mean(squeeze(img135fix(jj,50:462,50:462)),'all');
    stdev135fix(jj) = std2(squeeze(img135fix(jj,50:462,50:462)));
    
end
%% Plot Malus Law Comparison

errorbar(average0,stdev0,'*:');hold on; errorbar(average45,stdev45,'*:');errorbar(average90,stdev90,'*:');errorbar(average135,stdev135,'*:');legend('0 deg','45 deg','90 deg','135 deg');
figure;errorbar(average0fix,stdev0fix,'*:');hold on; errorbar(average45fix,stdev45fix,'*:');errorbar(average90fix,stdev90fix,'*:');errorbar(average135fix,stdev135fix,'*:');legend('0 deg','45 deg','90 deg','135 deg');xlabel('Input Angle (deg)');ylabel('Counts');

%% AoLP Uncorrected
clear AoLP
clear DoLP S1 S2 S0

for ii = 1:36
    S1(ii,:,:) = squeeze(img0(ii,50:462,50:462)) - squeeze(img90(ii,50:462,50:462));
    S2(ii,:,:) = squeeze(img45(ii,50:462,50:462)) - squeeze(img135(ii,50:462,50:462));

    S0(ii,:,:) = squeeze(img0(ii,50:462,50:462))./2 + squeeze(img45(ii,50:462,50:462))./2 + squeeze(img90(ii,50:462,50:462))./2 + squeeze(img135(ii,50:462,50:462))./2;

    DoLP(ii,:,:) = 100*(sqrt(S1(ii,246:266,246:266).^2 + S2(ii,246:266,246:266).^2)./S0(ii,246:266,246:266));
    DoLP_std(ii) = std2(100*(sqrt(S1(ii,246:266,246:266).^2 + S2(ii,246:266,246:266).^2)./S0(ii,246:266,246:266)));
    AoLP(ii) = mean(0.5*atan2(S2(ii,:,:),S1(ii,:,:)),'all');
    AoLP_std(ii) = std2(0.5*atan2(S2(ii,:,:),S1(ii,:,:)));
end

%% AoLP Corrected
clear AoLP
clear DoLP S1 S2 S0

for ii = 1:36
    S1(ii,:,:) = squeeze(img0fix(ii,50:462,50:462)) - squeeze(img90fix(ii,50:462,50:462));
    S2(ii,:,:) = squeeze(img45fix(ii,50:462,50:462)) - squeeze(img135fix(ii,50:462,50:462));

    S0(ii,:,:) = squeeze(img0fix(ii,50:462,50:462))./2 + squeeze(img45fix(ii,50:462,50:462))./2 + squeeze(img90fix(ii,50:462,50:462))./2 + squeeze(img135fix(ii,50:462,50:462))./2;

    DoLP(ii,:,:) = 100*(sqrt(S1(ii,246:266,246:266).^2 + S2(ii,246:266,246:266).^2)./S0(ii,246:266,246:266));
    DoLP_std(ii) = std2(100*(sqrt(S1(ii,246:266,246:266).^2 + S2(ii,246:266,246:266).^2)./S0(ii,246:266,246:266)));
    AoLP(ii) = mean(0.5*atan2(S2(ii,:,:),S1(ii,:,:)),'all');
    AoLP_std(ii) = std2(0.5*atan2(S2(ii,:,:),S1(ii,:,:)));
end
 %%
AoLPdeg = AoLP.*180/pi;
AoLPdeg(36) = AoLPdeg(36) - 180;
AoLP_std_deg = AoLP_std*180/pi;
AoLPdeg(9:17) = AoLPdeg(9:17) + 180;
AoLPdeg(27:36) = AoLPdeg(27:36) + 180;

input = 10:10:370;
input2 = 0:10:180;
input = input(1:36);
input(1:17) = input(1:17);
input(18:36) = input2(1:19);
input(36) = input(36)-180;

figure;errorbar(DoLP(1:36),DoLP_std(1:36),'.','MarkerSize',20);hold on;legend('DoLP measured');xlabel('GT angle (deg)');ylabel('DoLP (%)');set(gca,'FontSize',15);
set(gca,'XTick',(0:2:36),'XTickLabel',repmat(0:20:380,1, 20))%set new tick labels
figure;errorbar(AoLPdeg,AoLP_std_deg,'.','MarkerSize',20);hold on; plot(input,'*:','MarkerSize',10);legend('AoLP measured','AoLP input');xlabel('GT angle (deg)');ylabel('AoLP (deg)');set(gca,'FontSize',15);
set(gca,'XTick',(0:2:36),'XTickLabel',repmat(0:20:360,1, 20))%set new tick labels

%% Cos^2 fit
close all
xdata = 1:36;

myfun = @(x,xdata)(x(1).*cos(x(2).*xdata+x(3)).^2) + x(4);
NRCF = @(x) norm(Gpixel135 - myfun(x,xdata));
[X, ResNorm] = fminsearch(NRCF,[10000,20,1,4000]);


plot((X(1).*cos(X(2).*xdata+X(3)).^2) + X(4),'*:');hold on; plot(Gpixel135,'o');legend('fit','data');xlabel('Deg/10');ylabel('counts');