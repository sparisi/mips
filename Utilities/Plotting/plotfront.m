function plotfront(points, marker)
% PLOTFRONT Plots frontiers. By default, the square marker is used for 2d
% fronts, the circle for 3d fronts.

dim = size(points,2);

f = pareto(points);

if dim == 2
    f = sortrows(f);
    if nargin == 1, marker = 's'; end
    h = plot(f(:,1), f(:,2), ['--' marker]);
elseif dim == 3
    if nargin == 1, marker = 'o'; end
    h = plot3(f(:,1), f(:,2), f(:,3), marker);
    box on
    view(85,32)
else
    warning('Can plot only 2 and 3 dimensions.')
    return
end

set(h, 'MarkerSize', 7, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', h.Color);

alpha(0.7);
