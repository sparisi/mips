function h = pointsToSurf(x, y, z, nPoints_X, nPoints_Y, opacity)
% POINTSTOSURF Plots a surface from scattered data (X,Y,Z) with desired 
% OPACITY. NPOINTS_X and N_POINTS_Y define the density of the plot.

if nPoints_X < 1 || nPoints_Y < 1
    error('Less than 1 point selected.')
end

x_edge = linspace(floor(min(x)), ceil(max(x)), nPoints_X);
y_edge = linspace(floor(min(y)), ceil(max(y)), nPoints_Y);

[X,Y] = meshgrid(x_edge, y_edge);
Z = griddata(x,y,z,X,Y);

h = surf(X,Y,Z);
% h = mesh(X,Y,Z);

if nargin < 6
    opacity = 1;
end
alpha(h,opacity)

end