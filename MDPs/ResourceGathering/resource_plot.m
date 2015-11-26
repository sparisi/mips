function resource_plot(state, action)

persistent figureHandle agentHandle gemHandle goldHandle

[hasGems, hasGold, hasEnemy] = resource_environment;
cells = zeros(size(hasGems));
cells(5,3) = 1;

hasGold = flipud(hasGold)'; % Cartesian coord -> Matrix coord
hasGems = flipud(hasGems)';
hasEnemy = flipud(hasEnemy)';

if isempty(findobj('type','figure','name','Resource Plot'))
    figureHandle = figure();
    figureHandle.Name = 'Resource Plot';
    hold all
    
    h = imagesc(flipud(cells)); % Plot cells
    imggrid(h,'k',0.5); % Add grid
    colormap([1 1 1; 0.7 0.7 0.7])
    
    [x,y] = find(hasGold);
    goldHandle = plot(x,y,'ko','MarkerSize',24,'MarkerFaceColor','y'); 
    
    [x,y] = find(hasGems);
    gemHandle = plot(x,y,'kdiamond','MarkerSize',24,'MarkerFaceColor','m'); 
    
    [x,y] = find(hasEnemy);
    plot(x, y, 'kx', 'MarkerSize', 35, 'LineWidth', 15); % Enemies
    
    axis off
    
    agentHandle = plot(3,1,'ro','MarkerSize',10,'MarkerFaceColor','r'); % Submarine init
end

if nargin == 0
    return
end

[nrows, ncols] = size(resource_environment);
convertY = -(-nrows:-1); % Cartesian coord -> Matrix coord

x = state(2); % (X,Y) -> (Y,X)
y = state(1);

agentHandle.XData = x;
agentHandle.YData = convertY(y);

if state(3) % Carrying gems
    gemHandle.XData = x+0.3;
    gemHandle.YData = convertY(y)+0.3;
    gemHandle.MarkerSize = 5;
else
    [x,y] = find(hasGems);
    gemHandle.XData = x;
    gemHandle.YData = y;
    gemHandle.MarkerSize = 24;
end

if state(4) % Carrying gold
    goldHandle.XData = x+0.3;
    goldHandle.YData = convertY(y)+0.1;
    goldHandle.MarkerSize = 5;
else
    [x,y] = find(hasGold);
    goldHandle.XData = x;
    goldHandle.YData = y;
    goldHandle.MarkerSize = 24;
end

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
