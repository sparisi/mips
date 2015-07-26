function fig = plot3dPlane(pointA, pointB, pointC, opacity)

normal = cross(pointA-pointB, pointA-pointC); % calculate plane normal

% transform points to x, y, z
x = [pointA(1) pointB(1) pointC(1)];
y = [pointA(2) pointB(2) pointC(2)];
z = [pointA(3) pointB(3) pointC(3)];

% find all coefficients of plane equation
A = normal(1); B = normal(2); C = normal(3);
D = -dot(normal,pointA);

% decide on a suitable showing range
xLim = [min(x) max(x)];
zLim = [min(z) max(z)];
[X,Z] = meshgrid(xLim,zLim);
Y = (A * X + C * Z + D)/ (-B);
reOrder = [1 2 4 3];

fig = figure(); patch(X(reOrder),Y(reOrder),Z(reOrder),'b');
grid on;

if nargin == 3
    alpha(0.3);
else
    alpha(opacity);
end
