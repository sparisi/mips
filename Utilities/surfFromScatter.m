function surfFromScatter(x, y, z)

figure
grid on

tri = delaunay(x,y);

trisurf(tri, x, y, z);

light('Position',[-50 -15 29]);
lighting phong
shading interp
