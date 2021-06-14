% difference over sum of S01 & S02
%1708 = -227
%1513 = -678
%1534 = -309
%1535 = -435
%1539 = -99
%1541 = -379
%1543 = -499
%1544 = -376
%1546 = -144
%1547 = -497
%1549 = -166.5
%1550 = -9780
%1556 = -349
%1558 = -101
%1600 = -93
%1601 = 18
%1603 = 33
%1607 = -226
%1609 = -384
%1615 = -503
%1617 = -185
%1618 = -38
%1619 = -73
%1621 = -540
%1622 = -472.22
%1624 = -166
%1630 = 149
%1631 = -135
%1633 = -148
%1636 = 4.5
%1638 = -123
%1639 = 52
%1643 = 187
%1644 = -197
%1647 = -255
%1648 = -105
%1650 = -337
%1653 = -199
%1653may = 76
%1659 = -306
%1700 = -138
%1701 = -164
%1702 = -8000
%1704 = -279
%1706 = -306
%1709 = -147

testimage = load('clouds_1449_n420926_2375512.mat').image;

range = 1:512;
img0 = squeeze(testimage(1,range,range));
img45 = squeeze(testimage(2,range,range));
img90 = squeeze(testimage(3,range,range));
img135 = squeeze(testimage(4,range,range));


darkfield = load('darkfield_013secexp.mat').darkfield;
darkfield = squeeze(darkfield(1,:,:));

%pics for these
img0 = img0 - darkfield;
img45 = img45 - darkfield;
img90 = img90 - darkfield;
img135 = img135 - darkfield;

%Don't use caxis for these measurements

%Use caxis for these measurements
S0 = img0./2 + img90./2 + img45./2 + img135./2;
S01 = img0 + img90;
S02 = img45 + img135;
% S02test = img45 + (S01-img45);
img135test = 2.*(S01 - img0./2 + img90./2 + img45./2);
S02test = img45 + img135test;

S0diff = S01 - S02;

S0Sum = S01 + S02;

DiffSum = S0diff./S0Sum;

figure;imagesc(S0diff);colorbar;title('S0 diff');
figure;imagesc(DiffSum);axis off;colormap(gwp);colorbar;set(gca,'FontSize',15);caxis([-abs(max(DiffSum(:))) abs(max(DiffSum(:)))]);%title('S0a - S0b / S0a + S0b'); 

mean(S0diff(:))
mean(DiffSum(:))

max(DiffSum(:))-min(DiffSum(:))

%%
S0diff = S01 - S02test;

S0Sum = S01 + S02test;

DiffSum = S0diff./S0Sum;

figure;imagesc(S0diff);colorbar;title('S0 diff test');
figure;imagesc(DiffSum);colorbar;title('S0a - S0b / S0a + S0b test'); 

mean(S0diff(:))
max(S0diff(:))-min(S0diff(:))
mean(DiffSum(:))