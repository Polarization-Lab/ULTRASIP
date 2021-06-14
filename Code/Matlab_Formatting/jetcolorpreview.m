im = preview(vid);
ax = im.Parent;
% Specify scaled grayscale data mapping to colormap
im.CDataMapping = 'scaled';
% Specify a colormap to display grayscale image mapped to RGB image
colormap(ax, jet);
% Specify auto detection of CData limits
ax.CLimMode = 'auto';
% Or, specify a fixed signal data range to display
% signalRange = [10000 20000];
% ax.CLim = signalRange;