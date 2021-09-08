% AlignmentTest - Tests polarizer alignment by checking the slope of an
% average counts vs exposure data set.  DOES NOT WORK DO NOT USE
%
% Written by Atkin Hyatt 07/14/2021
% Last modified by Atkin Hyatt 07/14/2021

%initializeUV
C = zeros(1, 50); M = zeros(1, 50);
count = 0;
target = 1000;
image = zeros(1,512,512);
fopen(ELL14);

m0 = sample(vid,src,framesPerTrigger); m = m0; M(1) = m0;
fprintf('slope = %f\n', m)
rate = 0.1;

fprintf(ELL14, '%s', dec2hex(round(398.222222 * rate),8));    % step forward

while m >= target
    count = count + 1; C(count) = count;
    fprintf('Test %d', count);
    
    m = sample(vid,src,framesPerTrigger); M(count) = m;
    fprintf('slope = %f\n', m)
    
    diff = (m - target);
    if diff <= 0
        break
    end
    rate = diff * 0.005;
    
    if m > m0
        rate = rate * -1;
    end
    fprintf('%f\n', rate);
    fprintf(ELL14, "0mr" + dec2hex(round(398.2222222 * rate)));
    m = m0;
end
fclose(ELL14);

homeDeg = TranslateELL14(query(ELL14, '0gp'));
homeHex = dec2hex(round(398.2222222 * homeDeg));

fprintf('Home (deg) = %f\nHome (hex) = %f\n', homeDeg, homeHex);

function m = sample(vid,src,framesPerTrigger)
    src.ExposureTime = 0.1;
    image1 = UV_data(vid,framesPerTrigger);
    im1 = mean(mean(image1));
    
    src.ExposureTime = 1;
    image2 = UV_data(vid,framesPerTrigger);
    im2 = mean(mean(image2));
    
    m = (im1 - im2) / (0.1 - 1);
end