fig = openfig('UV_response.fig');

axObjs = fig.Children;
dataObjs = axObjs.Children;

x = dataObjs(1).XData;
y = dataObjs(1).YData;

y = y./max(y);

y = 0.1*1./y ; %scale factor

waves =  x(11:2:72);
exposures =  round(y(11:2:72),2);
