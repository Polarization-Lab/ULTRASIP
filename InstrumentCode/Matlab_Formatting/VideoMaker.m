for ii = 1:31
    figure(1);imagesc(squeeze(DoLP(ii,:,:)));colorbar;colormap(bone);set(gca,'FontSize',15);axis off;title('DoLP');%max(abs(DoLP(:)))]);
    title(['GT DoLP Scan = ' num2str(ii)])
    
    F(ii) = getframe(gcf);
    close all;
end

writerObj = VideoWriter('DoLPGT_nocorr','MPEG-4');
writerObj.FrameRate = 1;

open(writerObj);
for i = 1:length(F)
    frame = F(i);
    writeVideo(writerObj,frame);
end
close(writerObj)