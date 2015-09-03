function surfFromScatter(x, y, z)
% SURFFROMSCATTER Plots a surface from scattered data (X,Y,Z).

figure
grid on

tri = delaunay(x,y);

trisurf(tri, x, y, z);

light('Position',[-50 -15 29]);
lighting phong
shading interp
