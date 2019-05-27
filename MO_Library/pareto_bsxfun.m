function [s, p, idxs] = pareto_bsxfun(s, p)
% Like PARETO, but it uses only BSXFUN. It is usually slower.

[i, dim] = size(s);

if nargin == 1, p = zeros(i,1); end

idxs = sum(bsxfun(@le, s, permute(s, [3,2,1])), 2) == dim;
idxs = sum(idxs,3) == 1;
p = p(idxs,:);
idxs = find(idxs);
