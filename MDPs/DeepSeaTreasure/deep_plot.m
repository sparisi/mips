function deep_plot(state, action)

persistent figureHandle agentHandle

[treasure, cells] = deep_environment;
cells = cells * 1; % From logical to double
cells(treasure > 0) = 0.5;
cells = flipud(cells);
nrows = size(deep_environment,1);
convertY = -(-nrows:-1); % Cartesian coord -> Matrix coord

if isempty(findobj('type','figure','name','Deep Plot'))
    figureHandle = figure();
    figureHandle.Name = 'Deep Plot';
    hold all
    h = image(cells); % Plot environment
    colormap([0 0 0; 0.5 0.5 0.5; 1 1 1]);
    imggrid(h,'k',0.5); % Add grid
    
    treasure = flipud(treasure)'; % Cartesian coord -> Matrix coord
    [rows,cols] = find(treasure);
    for i = 1 : length(rows) % Add treasures value
        text('position', [rows(i) cols(i)], ...
            'fontsize', 10, ...
            'string', num2str(treasure(rows(i),cols(i))), ...
            'color', 'white', ...
            'horizontalalignment', 'center')
    end
    
    axis off
    agentHandle = plot(1,nrows,'ro','MarkerSize',8,'MarkerFaceColor','r'); % Submarine init
end

if nargin == 0
    return
end

x = state(2); % (X,Y) -> (Y,X)
y = state(1);

agentHandle.XData = x;
agentHandle.YData = convertY(y);

title('')
if nargin == 2
    switch action
        case 1, str = 'Left';
        case 2, str = 'Right';
        case 3, str = 'Up';
        case 4, str = 'Down';
    end
    title(str)
end

end
