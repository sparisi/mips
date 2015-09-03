function hv = hypervolume2d(f, antiutopia, utopia)
% HYPERVOLUME2D Computes the hypervolume of a 2-dimensional frontier for a 
% maximization problem. If the user provides both the utopia and antiutopia,  
% the frontier is normalized in order to have the objectives in the range 
% [0, 1]. If only the antiutopia point is provided, the frontier is not 
% normalized and the antiutopia is chosen as reference point.
% Please note that points at the same level or below the antiutopia are not
% considered, so choose the antiutopia carefully. For example, if the 
% antiutopia is [0,-19], the point [124, -19] is ignored, so it would be
% better to choose [124, -20] as antiutopia.
%
%    INPUT
%     - f          : N-by-D matrix representing a D-dimensional Pareto 
%                    front of N points
%     - antiutopia : antiutopia point (1-by-D vector)
%     - utopia     : utopia point (1-by-D vector)
%
%    OUTPUT
%     - hv         : hypervolume

if nargin == 3
    f = normalize_points(f,antiutopia,utopia);
    r = zeros(1, size(f,2));
else
    r = antiutopia;
end

% If a solution lays below the reference point, ignore it
isBelow = sum( bsxfun(@gt, r, f), 2) >= 1;
f(isBelow, :) = [];

if isempty(f)
    hv = 0;
    return
end

f = sortrows(f,1);
b = f(1,1) - r(1);
h = f(1,2) - r(2);
hv = b*h;
for i = 2 : size(f,1)
    b = f(i,1) - f(i-1,1);
    h = f(i,2) - r(2);
    hv = hv + b*h;
end

return
