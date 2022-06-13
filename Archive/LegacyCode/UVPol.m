ASM = NET.addAssembly('C:\UV Polarimeter Computer Backup\ESP301 .NET Assembly\.NET Assembly\Newport.ESP301.CommandInterface.dll'); %Call DLL
%%
ESP301 = CommandInterfaceESP301.ESP301(); %Create Instance
%%
err1 = ESP301.OpenInstrument('COM6',921600); %Open USB connection
if((err1) == 0)
    disp('No error connecting to the ESP301');
end
%%
[error,Version] = ESP301.VE(); % Get Version 

disp(Version); %Display Version
%%
err2 = ESP301.CloseInstrument(); %Close USB Connection

