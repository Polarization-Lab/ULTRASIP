% ULTRASIP_IntegratingSphere_Analysis

addpath('C:\ULTRASIP_Data\July2021\Uncorrected Data');

f = filename;
savedir = 'C:\ULTRASIP_Data\July2021\Corrected Data\';

% Data from first scan
[correctData1, DoLP, AoLP] = IntegratingSphere_Correction(f, 1);

D1 = mean(mean(DoLP));
A1 = mean(mean(AoLP));

% Data from second scan
[correctData2, DoLP, AoLP] = IntegratingSphere_Correction(f, 2);

D2 = mean(mean(DoLP));
A2 = mean(mean(AoLP));

% Save data
correctImage = [correctData1; correctData2];