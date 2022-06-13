function [] = movePSG(xps,newPos)
%MOVEPSG this function changes the position of the PSG
% xps is the initialized XPS instance  
% newPos is the new angular position

%round value
pos = round(newPos,4);


%define the positioners
PSG     = 'Group1' ;
PSGpos  =  'Group1.Pos';
PSA     = 'Group2' ;
PSApos  =  'Group2.Pos';


disp('Moving PSG')
%moves PSG and waits til finished
G1=xps.GroupMoveAbsolute(PSG,pos,1); %move psg
[G1,CurrPos1]=xps.GroupPositionCurrentGet(PSG,1); %get new psg position
while double(CurrPos1) ~= pos %Keep recording position while PSG is moving
    pause(.1)
    [G1,CurrPos1]=xps.GroupPositionCurrentGet(PSG,1); 
    CurrPos1=double(CurrPos1);
end

end


