function s = StdDevCorrected(gamma0,image2correct,Avg_M,Avg_B,M,B)

C = ImgCorrection(image2correct,gamma0,M,B,Avg_M,Avg_B);

A = fftshift(fft2(fftshift(C)));
A_mod = abs(A);

x = -255:256;
y = -255:256;
[xx yy] = meshgrid(x,y);
mask = ones(size(xx));
mask((xx.^2+yy.^2)<100^2)=1;
mask((xx.^2+yy.^2)<5^2)=0;
mask = imgaussfilt(mask,20);
%figure(); colormap gray; imagesc(mask);
filtered_spectrum = mask.*A;
filtered_spectrum_mod = abs(filtered_spectrum);

%A = A(182:248,263:327);
s = std(filtered_spectrum_mod(:));