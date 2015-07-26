function hv = hypervolume2d(f, antiutopia, utopia)
% Computes the hypervolume of a 2-dimensional frontier for a maximization
% problem. The frontier is always normalized in order to have the
% objectives in the range [-1, 0] or [0, 1]. The normalization is performed
% using the utopia and antiutopia points. If only the antiutopia point is
% provided, the frontier is not normalized and the antiutopia is chosen as
% reference point.
%
% Inputs:
% - f          : N-by-D matrix representing a D-dimensional Pareto front of N
%                points
% - antiutopia : antiutopia point (1-by-D vector)
% - utopia     : utopia point (1-by-D vector)
%
% Outputs:
% - hv         : hypervolume
%
% Example: if the reference point is [-10, 10] then the frontier is
% normalized by r and the new reference point is [-1, 0].

% Find the correct factor to normalize the frontier in the interval [-1,0] (or [0, 1])
if nargin == 3
    m = [utopia; antiutopia];
    [~, idx] = max(abs(m));
    r = m([idx == 1; idx == 2])'; % Reference point

    f = f * diag( 1 ./ abs(r) ); % Normalize the objectives
    r = (antiutopia >= 0) - 1; % Change the reference point
else
    r = antiutopia;
end

% If a solution lays below the reference point, ignore it
isBelow = sum( bsxfun(@ge, r, f), 2) >= 1;
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
