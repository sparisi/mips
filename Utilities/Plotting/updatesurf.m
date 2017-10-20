function updatesurf(name, X, Y, newZ)
% UPDATESURF Updates surf plots without generating a new figure.
%
%    INPUT
%     - name : figure name
%     - X    : X as in surf
%     - Y    : Y as in surf
%     - newZ : Z as in surf

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

% If the figure does not exist, create it and plot the surf
if isempty(fig)
    fig = figure();
    fig.Name = name;
    surf(X,Y,newZ)
    title(name)
    return
end

% Update Z values
dataObj = findobj(fig,'Type','Surface');
set(dataObj, 'CData', newZ);
set(dataObj, 'ZData', newZ);

drawnow limitrate
