function p = linspacesim(a, b, n)
% LINSPACESIM Like LINSPACE, but points are sampled from the simplex.
%
%    INPUT
%     - a : lower bound, vector of length D
%     - b : upper bound, vector of length D
%     - n : number of samples
%
%    OUTPUT
%     - p : [D x N] matrix with random points, where N is the closest 
%           integer to n satisfying the simplex sampling conditions
%
% =========================================================================
% More information about sampling from the simplex:
% http://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
%
% It requires 'inhull':
% http://www.mathworks.com/matlabcentral/fileexchange/10226-inhull

if ~iscolumn(a), a = a'; end
if ~iscolumn(b), b = b'; end
assert(min(a <= b) == 1, 'Bounds are not consistent.')

dim = length(a);

if dim == 1
    p = linspace(a,b,n);
    return
end

linspaces = cell(dim,0);

% When using the simplex, the volume of the hypercuboid defined by
% [lo,hi] is reduced by a factor DIM!. This explains why we need to
% increase n to guarantee the desired number of samples.
n = n * factorial(dim);

for i = 1 : dim
    linspaces{i} = linspace(a(i),b(i),floor(nthroot(n,dim)));
end
c = cell(1,dim);
[c{:}] = ndgrid(linspaces{:});
p = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput', false) )';

pp = zeros(dim+1, dim);
for i = 1 : dim
    tmp = a';
    tmp(i) = b(i);
    pp(i,:) = tmp;
end
pp(end,:) = a';
idx = inhull(p',pp);
p = p(:,idx);
