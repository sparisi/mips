function out = trapz2d(f, x_min, x_max, y_min, y_max, n_points)
% Double trapezoidal integral over a uniformly spaced interval.
%
% Inputs:
%  - f        : handle to the (vectorial) function to integrate
%  - x_min    : lower bound of the first variable
%  - x_max    : upper bound of the first variable
%  - y_min    : lower bound of the second variable
%  - y_max    : upper bound of the second variable
%  - n_points : number of points to sample per dimension
% 
% Outputs:
%  - out      : (vector column) result

n = ceil(sqrt(n_points));
x = linspace(x_min, x_max, n);
y = linspace(y_min, y_max, n);

[xx,yy] = meshgrid(x,y);
z = f(xx,yy);
if size(z,2) < size(z,1)
    z = z';
end
z = reshape(z,n,n,numel(z)/n^2);
out = trapz(y,trapz(x,z,1),2);
out = out(:);

end
