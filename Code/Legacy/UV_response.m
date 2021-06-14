waves = 400:5:800;
response = zeros(1,81);

for i = 1:81
    changeWavelength(COMmono,waves(i))
    pause(2)
    start(vid)
    im = getdata(vid);
    response(i) = mean(im,'all');
    stop(vid)
end

plot(waves,response,'-.')
title('Instrument Response')
xlabel('Wavelength [nm]')
ylabel('Avg. Camera Count')
grid('on')