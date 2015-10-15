hold all

[hasGems, hasGold, hasEnemy] = resource_environment;
cells = zeros(size(hasGems));
cells(5,3) = 1;

h = imagesc(flipud(cells)); % Plot cells
imggrid(h,'k',0.5); % Add grid
colormap([1 1 1; 0.7 0.7 0.7])

hasGold = flipud(hasGold)'; % Cartesian coord -> Matrix coord
[x,y] = find(hasGold);
hGold = circles(x, y, 0.3, 'color', 'y', 'edgecolor', 'k'); % Gold

hasGems = flipud(hasGems)';
[x,y] = find(hasGems);
hGems = circles(x, y, 0.3, 'vertices', 4, 'rot', 0, 'color', 'm', 'edgecolor', 'k'); % Gems

hasEnemy = flipud(hasEnemy)';
[x,y] = find(hasEnemy);
plot(x, y, 'kx', 'MarkerSize', 35, 'LineWidth', 15); % Enemies

axis off