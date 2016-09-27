function refreshplot(name, newX, newY, newZ)
% REFRESHPLOT Refresh 2d and 3d plots with new data.
%
%    INPUT
%     - name     : figure name
%     - newX     : array of new XData
%     - newY     : array of new YData
%     - newZ     : array of new ZData

fig = findobj('type','figure','name',name);

if isempty(fig), error('Figure not found.'), end

axes = findobj(fig,'type','axes');
axes.Children.XData = newX;
axes.Children.YData = newY;
axes.Children.ZData = newZ;
    
drawnow limitrate
