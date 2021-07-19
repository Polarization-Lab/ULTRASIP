function img = ImgCorrection(image2correct,gamma0,M,B,Avg_M,Avg_B)

% flattest = load('FPN_flatfieldSys.mat').flat;
% 
% flattest = flattest(2:26,:,:);
%     %%
% clear m pixelarray B
% u = 1:24;
% for ii = 1:512
%     for jj = 1:512
%         for uu = 1:24
%             pixelarray(uu) = flattest(uu,ii,jj);
%         end
%         x = [ones(length(u),1) u'];
%         var = x\pixelarray';
%         M(ii,jj) = var(2);
%         B(ii,jj) = var(1);
%     end
% end
% 
% %%
% %Rc(ij) = Rij + (1-M*/Mij)(Rij-Bij) + (Bij - B*)
% %M*
% Avg_M = mean(M(:));
% %B*
% Avg_B = mean(B(:));
%% Load Image

for ii = 1:512
    for jj = 1:512
        %imgCorrect(ii,jj) = (image2correct(ii,jj) - B(ii,jj))*(1-gamma0*(1-Avg_M/M(ii,jj)))+Avg_B;
        imgCorrect(ii,jj) = (image2correct(ii,jj))*(gamma0*(Avg_M))+Avg_B;

    end
end



img = imgCorrect;

%%
% for ii = 1:512
%     for jj = 1:512
%         imgCorrector(ii,jj) = (Avg_M/M(ii,jj))*(B(ii,jj)-alpha*image2correct(ii,jj))+ alpha*image2correct(ii,jj)-Avg_B;
%         imgCorrect(ii,jj) = image2correct(ii,jj) - imgCorrector(ii,jj);
%     end
% end
% 
% 
% 
% img = imgCorrect;
%%
% % image2correct = img0;
% 
% imgCorrect = zeros(512,512);
% 
% 
% for ii = 1:512
%     for jj = 1:512
%         imgCorrector(ii,jj) = alpha*(1-Avg_M/M(ii,jj))*(image2correct(ii,jj)-B(ii,jj)) + (B(ii,jj)-Avg_B);
%         imgCorrect(ii,jj) = image2correct(ii,jj) - imgCorrector(ii,jj);
%     end
% end
% 
% % figure;imagesc(image2correct);title('OG');colorbar;%caxis([0 38000]);
% % figure;imagesc(imgCorrector*3.5);colormap(gwp);caxis([-max(abs(imgCorrector(:))) max(abs(imgCorrector(:)))]);title('Corrector');colorbar;
% % figure;imagesc(imgCorrect);title('Corrected'); colorbar;
% 
% img = imgCorrect;
end