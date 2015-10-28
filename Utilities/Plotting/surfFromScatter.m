function h = surfFromScatter(x, y, z, opacity)
% SURFFROMSCATTER Plots a surface from scattered data (X,Y,Z) with desired
% OPACITY.

tri = delaunay(x,y);

h = trisurf(tri, x, y, z);

light('Position',[-50 -15 29]);
lighting phong
shading interp

if nargin < 4
    opacity = 1;
end
alpha(h,opacity)