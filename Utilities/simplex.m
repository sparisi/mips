function [s, v] = simplex(a, b)
% SIMPLEX Returns the simplex and the volume of an interval [A,B].
%
%    INPUT
%     - a : lower bound, vector of length D
%     - b : upper bound, vector of length D
%
%    OUTPUT
%     - s : indices of the points in [A,B] that comprise the facets of the 
%           simplex
%     - v : volume of the simplex


if ~iscolumn(a), a = a'; end
if ~iscolumn(b), b = b'; end
assert(min(a <= b) == 1, 'Bounds are not consistent.')
assert(length(a) == length(b), 'Bounds must have the same length')

d = length(a);

if d == 1
    v = 1;
    s = [a b];
else
    pp = zeros(d+1, d);
    for i = 1 : d
        p = a';
        p(i) = b(i);
        pp(i,:) = p;
    end
    pp(end,:) = a';
    [s, v] = convhulln(pp);
end
