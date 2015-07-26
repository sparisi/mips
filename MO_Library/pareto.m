function [ s, p, idxs] = pareto( s, p )
% Filters a set of points S according to Pareto dominance, i.e., points
% that are dominated (both weakly and strongly) are filtered.
%
% Inputs:
% - S    : N-by-D matrix, where N is the number of points and D is the
%          number of elements (objectives) of each point.
% - P    : N-by-D matrix containing the policies that generated S
%
% Outputs:
% - S    : Pareto-filtered S
% - P    : Pareto-filtered P
% - idxs : indices of the non-dominated solutions
%
% Example:
% s = [1 1 1; 2 0 1; 2 -1 1; 1, 1, 0];
% [f, ~, idxs] = pareto(s)
%     f = [1 1 1; 2 0 1]
%     idxs = [1; 2]

if nargin == 1
    p = zeros(size(s));
end
[i, dim] = size(s);
idxs = [1 : i]';
while i >= 1
    old_size = size(s,1);
    indices = sum( bsxfun( @ge, s(i,:), s ), 2 ) == dim;
    indices(i) = false;
    s(indices,:) = [];
    p(indices) = [];
    idxs(indices) = [];
    i = i - 1 - (old_size - size(s,1)) + sum(indices(i:end));
end

end