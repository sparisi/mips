function updatescatter(name, X, Y, Z, value)
% UPDATESCATTER Updates scatter plots without generating a new figure.
% The call is scatter(X,Y,1,value) or scatter3(X,Y,Z,1,value) if Z is not
% empty.

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

% If the figure does not exist, create it and plot the surf
if isempty(fig)
    fig = figure();
    fig.Name = name;
    if isempty(Z), scatter(X,Y,1,value),
    else, scatter3(X,Y,Z,1,value), end
    title(name)
    return
end

% Update Z values
dataObj = findobj(fig,'Type','Scatter');
set(dataObj, 'XData', X);
set(dataObj, 'YData', Y);
set(dataObj, 'ZData', Z);
set(dataObj, 'CData', value);

drawnow limitrate
