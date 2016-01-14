function fig = plot3dPlane(pointA, pointB, pointC, color, opacity, fig)
% PLOT3DPLANE Plots a 3d plane passing through three points. If given, the
% plane is plotted in the figure FIG. Otherwise, a new figure is returned.

normal = cross(pointA-pointB, pointA-pointC);

d = -pointA * normal';

x = [pointA(1) pointB(1) pointC(1)];
y = [pointA(2) pointB(2) pointC(2)];
[xx, yy] = ndgrid(x,y);

z = (-normal(1) * xx - normal(2) * yy - d) / normal(3);

if nargin < 6, fig = figure();
else figure(fig); end

hsurf = surf(xx,yy,z);
set(hsurf,'FaceColor',color,'FaceAlpha',opacity);
grid on
