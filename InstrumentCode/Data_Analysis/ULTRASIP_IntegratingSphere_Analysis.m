% ULTRASIP_IntegratingSphere_Analysis -- Correct polishing marks and
% extract DoLP and AoLP position from integrating sphere experiments
%
% Written by Atkin Hyatt 08/12/2021
% Last modified by Atkin Hyatt 08/12/2021

addpath('C:\ULTRASIP_Data\Data2021\Uncorrected Data');
addpath('C:\ULTRASIP_Data\Data2021\Corrected Data');

savedir = 'C:\ULTRASIP_Data\Data2021\Corrected Data\';

%% Fix data
clear stdevDOLP stdevAOLP stdevData
correctedImage = zeros(4*iter,323,275); stdevDOLP = zeros(1,iter); stdevAOLP = zeros(1,iter);
for ii = 1 : iter
    fprintf("Correcting scan %d", ii)
    [correctData, DoLP, AoLP, S0, S1, S2] = IntegratingSphere_Correction(filename, ii, "file");
    
    correctedImage((4*ii-3) : (4*ii), :, :) = correctData;
    stdevData(1,ii) = std(reshape(DoLP, 1, 323*275));
    stdevData(2,ii) = std(reshape(AoLP, 1, 323*275));

    S(3*ii-2,:,:) = S0; S(3*ii-1,:,:) = S1; S(3*ii,:,:) = S2; 
    
    rawData(1,ii,:,:) = DoLP; rawData(2,ii,:,:) = AoLP;
end

%% Save data to computer
f = [savedir '' file];
% create new branch for calculated data
h5create(f,'/measurement/polarization/radiometric',size(correctedImage),"Chunksize",[4*iter 323 275]);
h5create(f,'/measurement/polarization/polarizationmetric',size(rawData),"Chunksize",[2 iter 323 275]);
h5create(f,'/measurement/polarization/error',size(stdevData),"Chunksize",[2 iter]);
h5create(f,'/measurement/polarization/stokes',size(S),"Chunksize",[3*iter 323 275]);
h5create(f,'/measurement/polarization/datapoints',size(iter),"Chunksize", size(iter));
h5create(f,'/measurement/polarization/exposuretime', size(expo),"Chunksize", size(expo));

% Write data to branch
h5write(f,'/measurement/polarization/radiometric/',correctedImage);
h5write(f,'/measurement/polarization/polarizationmetric/',rawData);
h5write(f,'/measurement/polarization/error',stdevData);
h5write(f,'/measurement/polarization/stokes',S);
h5write(f,'/measurement/polarization/datapoints',iter);
h5write(f,'/measurement/polarization/exposuretime',expo);