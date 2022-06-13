triggerconfig(vid, 'manual');
disp('Taking data')

%Start camera comms
start(vid)

%Check if ESP301 is open
if ~isempty(instrfind(esp301,'Status','close'))
    fopen(esp301);
end

%Take images
% tic
for ii = 1:721
%     tic   
    fprintf('Taking data %0.3f Degrees\n',ii);  

    fprintf(esp301,sprintf('1PA%0.3f',ii));
    %different loop values ^    

    query(esp301,sprintf('1WS;1TP?')) %ask position

    image(ii,:,:) = UV_data(vid,framesPerTrigger); %take picture
    Intensity(ii) = max(image(ii,:));
    %pause(2);
    disp('image taken')
    
%     eachtime(ii) = toc;
end
% finaltime = toc;

%Close instruments
fclose(esp301);
stop(vid)
disp('Done')

plot(Intensity)