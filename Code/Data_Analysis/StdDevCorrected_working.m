function s = StdDevCorrected_working(gamma,image2correct,Avg_M,Avg_B,M,B)
%C = image2correct;
C = ImgCorrection_working(image2correct,gamma,M,B,Avg_M,Avg_B);

A = fftshift(fft2(fftshift(C)));
A_mod = abs(A);

 x = -255:256;
 y = -255:256;
 [xx yy] = meshgrid(x,y);
 mask = ones(size(xx));
 %mask((xx.^2+yy.^2)<20^2)=1;
 %mask((xx.^2+yy.^2)<6^2)=0;
 % mask((xx.^2+yy.^2)<200^2)=1;
 % mask((xx.^2+yy.^2)<25^2)=0;
mask(((xx-1).^2+(yy-1).^2)<20^2)=1;
 mask(((xx-1).^2+(yy-1).^2)<6^2)=0;
% %mask = imgaussfilt(mask,20);
 figure(2); colormap gray; imagesc(mask);title('Annular Mask','FontSize',20);colorbar('FontSize',20);
 filtered_spectrum = mask.*A;
 filtered_spectrum_mod = abs(filtered_spectrum);
 figure (3);colormap copper;imagesc(filtered_spectrum_mod);title('Annular Mask for Standard Deviation ROI','FontSize',16);colorbar('FontSize',16);
  %figure (1);colormap gray;imagesc(A_mod);colorbar;

%A = A(182:248,263:327);
 s = std(filtered_spectrum_mod(:));
%s = std(A(:,:));

disp(s)
end