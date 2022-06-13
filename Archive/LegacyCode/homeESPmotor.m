function [] = homeESPmotor()
%HOMEMOTOR Summary of this function goes here
%James Heath heathjam@email.arizona.edu
%Sept 25 2020
%This script is designed to home ESP motor

%define the positioners
PSG     = 'Group1' ;
PSGpos  =  'Group1.Pos';
PSA     = 'Group2' ;
PSApos  =  'Group2.Pos';

%Group Kill
 xps.GroupKill(PSG);
 xps.GroupKill(PSA);


%initailizes stages or 'Groups'
G1=xps.GroupInitialize(PSG); %PSG rotates slower
G2=xps.GroupInitialize(PSA); %PSA rotates faster

disp('Homing PSG')
%Homes and updates Current Position of Stages
G1=xps.GroupHomeSearch(PSG); %PSG Homes to abs zero
[G1,CurrPos1]=xps.GroupPositionCurrentGet(PSG,1);    %PSA Homes to abs zero

%Updates current position and is a wait func til PSG are done homing
while double(CurrPos1) ~= 0
    pause(.1)
    [G1,CurrPos1]=xps.GroupPositionCurrentGet(PSG,1); %Grabs current Pos, 
    CurrPos1=double(CurrPos1); %output system array to matlab array (See .NET Array)
end

disp('Homing PSA')
%homes PSA and waits til finished
G2=xps.GroupHomeSearch(PSA);
[G2,CurrPos2]=xps.GroupPositionCurrentGet(PSA,1);
while double(CurrPos2) ~= 0
    pause(.1)
    [G2,CurrPos2]=xps.GroupPositionCurrentGet(PSA,1);
    CurrPos2=double(CurrPos2);
end

end

