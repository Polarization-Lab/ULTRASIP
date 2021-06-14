function [] = changeWavelengthUV(COMmono, wavelength)
%CHANGEWAVELENGTH this changes the wavelength of the monochromator
%   wavelength is the wavelengt hin nm, a double

cmd = join(['WAVE ',num2str(wavelength)]); %create serial command
fprintf(COMmono,cmd); %Sends current wavelength cmd, which is echoed
Mono_Status=fscanf(COMmono);  %reads echo
disp(Mono_Status)
Mono_Status=fscanf(COMmono);  %reads wavelength (str)
disp(Mono_Status)

end

