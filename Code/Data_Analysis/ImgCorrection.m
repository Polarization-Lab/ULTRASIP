function img = ImgCorrection(image2correct,gamma,M,B,Avg_M,Avg_B)
%% Load Image

for ii = 1:512
    for jj = 1:512
        imgCorrect(ii,jj) = (image2correct(ii,jj) - B(ii,jj))*(1-(gamma*(1-Avg_M/M(ii,jj))))+Avg_B;
   
    end
end

img = imgCorrect;


end