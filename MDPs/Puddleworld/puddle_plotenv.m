p1 = [0.1 0.75; % Centers of the first puddle
    0.45 0.75];
p2 = [0.45 0.4; % Centers of the second puddle
    0.45 0.8];

radius = 0.1;

hold all

% Circles
p = [p1; p2];
circles(p(:,1), p(:,2), radius, 'color', 'black', 'edgecolor', [1 1 1])

% Rectangles
patch([0.1 0.45 0.45 0.1], [0.65 0.65 0.85 0.85], 'black', 'EdgeAlpha', 0)
patch([0.35 0.55 0.55 0.35], [0.4 0.4 0.8 0.8], 'black', 'EdgeAlpha', 0)

alpha(0.6)

% Triangle
x = [0.95, 1.0, 1.0];
y = [1.0, 0.95, 1.0];
fill(x, y, 'r')

axis([0 1 0 1])
box on
axis square