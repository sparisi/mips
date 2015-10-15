[treasure, cells] = deep_environment;

hold all

cells = cells * 1; % From logical to double
cells(treasure > 0) = 0.5;

cells = flipud(cells);
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