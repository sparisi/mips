function updateplot(name, newX, newY)
% UPDATEPLOT Updates 2d plots with new points.
%
%    INPUT
%     - name : figure name
%     - newX : cell array of new single X-coord (one element per plot)
%     - newY : cell array of new single Y-coord (one element per plot)
%
% =========================================================================
% EXAMPLE
% for i = 1 : 10, updateplot('Test',{i,i},{rand,rand*10}), pause(0.5), end

% Look for a figure with the specified name
fig = findobj('type','figure','name',name);

% If the figure does not exist, create it and plot the first points
if isempty(fig)
    fig = figure();
    fig.Name = name;
    hold all
    for i = 1 : numel(newX)
        plot(newX{i}, newY{i})
    end
    hold off
    title(name)
    return
end

% Find plots in the figure
axesObjs = get(fig, 'Children');
dataObjs = get(axesObjs, 'Children');
X = get(dataObjs, 'XData')';
Y = get(dataObjs, 'YData')';
if ~iscell(X), X = {X}; Y = {Y}; end % If there is only one plot

% Append new points
for i = 1 : numel(dataObjs)
    X{i}(end+1) = newX{i};
    Y{i}(end+1) = newY{i};
    set(dataObjs(i), 'XData', X{i});
    set(dataObjs(i), 'YData', Y{i});
end

drawnow limitrate
