function s = StdDevCorrected(gamma0,image2correct,Avg_M,Avg_B,M,B)

C = ImgCorrection(image2correct,gamma0,M,B,Avg_M,Avg_B);

A = fftshift(fft2(C));
A = abs(A);
% A = A(182:248,193:249);
A = A(182:248,263:327);
s = std(A(:));