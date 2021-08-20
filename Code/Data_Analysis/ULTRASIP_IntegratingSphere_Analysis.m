% ULTRASIP_IntegratingSphere_Analysis -- Correct polishing marks and
% extract DoLP and AoLP position from integrating sphere experiments
%
% Written by Atkin Hyatt 08/12/2021
% Last modified by Atkin Hyatt 08/12/2021

addpath('C:\ULTRASIP_Data\July2021\Uncorrected Data');
addpath('C:\ULTRASIP_Data\July2021\Corrected Data');

savedir = 'C:\ULTRASIP_Data\July2021\Corrected Data\';

%% Fix data
% Data from first scan
[correctData1, DoLP1, AoLP1, S01, S11, S21] = IntegratingSphere_Correction(filename, 1, "file");

stdevDOLP(1) = std(reshape(DoLP1, 1, 323*275));
stdevAOLP(1) = std(reshape(AoLP1, 1, 323*275));

D1 = mean(mean(DoLP1));
A1 = mean(mean(AoLP1));

% Data from second scan
[correctData2, DoLP2, AoLP2, S02, S12, S22] = IntegratingSphere_Correction(filename, 2, "file");

stdevDOLP(2) = std(reshape(DoLP2, 1, 323*275));
stdevAOLP(2) = std(reshape(AoLP2, 1, 323*275));

D2 = mean(mean(DoLP2));
A2 = mean(mean(AoLP2));

% Save data
correctImage = [correctData1; correctData2]; 
stdevData = [stdevDOLP; stdevAOLP];
stokes(1,:,:) = S01; stokes(2,:,:) = S11; stokes(3,:,:) = S21;
stokes(4,:,:) = S02; stokes(5,:,:) = S12; stokes(6,:,:) = S22; 
correctData(1,:,:) = DoLP1; correctData(2,:,:) = DoLP2;
correctData(3,:,:) = AoLP1; correctData(4,:,:) = AoLP2;

%% Save data to computer
f = [savedir '' file];
% create new branch for calculated data
h5create(f,'/measurement/polarization/radiometric',size(correctImage),"Chunksize",[8 323 275]);
h5create(f,'/measurement/polarization/polarizationmetric',size(correctData),"Chunksize",[4 323 275]);
h5create(f,'/measurement/polarization/error',size(stdevData),"Chunksize",[2 2]);
h5create(f,'/measurement/polarization/stokes',size(stokes),"Chunksize",[6 323 275]);

% Write data to branch
h5write(f,'/measurement/polarization/radiometric/',correctImage);
h5write(f,'/measurement/polarization/polarizationmetric/',correctData);
h5write(f,'/measurement/polarization/error',stdevData);
h5write(f,'/measurement/polarization/stokes',stokes);