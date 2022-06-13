function [COMmono] = initializeMonoUV()
%INITIALIZEMONO Summary of this function goes here
%Kira Hart khart@optics.arizona.edu
%July 28 2020
%This script is designed to initialize the monochromator
%and display the current wavelength

disp('Initializing Monochromator')

COMmono=serialport("COM2",9600,'DataBits' , 8, 'Parity' , 'none' ,'StopBits' , 1,'FlowControl', 'none','Timeout', 3);  %init serial comm w/ Monochromator
% set( COMmono , 'BaudRate' , 9600 ,...
%      'DataBits' , 8 ,...
%      'Parity' , 'none' ,...
%      'StopBits' , 1,...
%      'FlowControl', 'hardware',...
%      'Terminator', 13 ,...
%      'Timeout', 3);
 configureTerminator(COMmono,13);
 
 %%
 setRTS(COMmono, true) 

 %%
 fopen(COMmono);           %opens comm/VISA w/ Mono, can now send commands
fprintf(COMmono,56); %Sends current wavelength cmd, which is echoed
fprintf(COMmono,'Current Wavelength');
Mono_Status=fscanf(COMmono);  %reads echo
Mono_Status=fscanf(COMmono);  %reads wavelength (str)
fclose(COMmono);

%%
disp('Wavelength is set at [nm]')
disp(Mono_Status)
Mono_Status=str2double(Mono_Status);    %converts to num

pause(2)

%%
Mono = serial('COM9');
set(Mono, 'baudrate', 9600);


%%
fopen(COMmono)
fprintf(COMmono,char(27)); %Sends current wavelength cmd, which is echoed
Mono_status=fscanf(COMmono)
fclose(COMmono)

%%
writeline(COMmono, char(27))
Mono_status = readline(COMmono);
%%
clear COMmono
clear Mono_status

end



