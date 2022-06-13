function s = StdDevCorrected(gamma,image2correct,Avg_M,Avg_B,M,B)

C = ImgCorrection(image2correct,gamma,M,B,Avg_M,Avg_B);

A = fftshift(fft2(fftshift(C)));
A_mod = abs(A);

%Annular Mask
x = -255:256;
y = -255:256;
[xx yy] = meshgrid(x,y);
mask = ones(size(xx));
mask(((xx-1).^2+(yy-1).^2)<20^2)=1;
mask(((xx-1).^2+(yy-1).^2)<6^2)=0;

filtered_spectrum = mask.*A;
filtered_spectrum_mod = abs(filtered_spectrum);
%figure (3);colormap gray;imagesc(log(filtered_spectrum_mod));axis off

s = std(filtered_spectrum_mod(:));

end