function hv = metric_hv_bsxfun(F, U, AU, N)
% Same as METRIC_HV, but always uses a Monte Carlo estimate of the
% hypervolume. It is generally faster, but may be slow if F or N are large.
% Also, it does not add a penalty and does not check for duplicates.

dim = length(U);
P = bsxfun( @plus, AU, bsxfun(@times, (U - AU), rand(N, dim)) ); % hypercuboid
tmp = permute(F, [3,2,1]);

idx = sum(bsxfun(@le, P, tmp), 2) == dim; % (i,j) says is point P(i) is dominated by F(j)
idx_unique = sum(idx, 3) == 1; % points in P dominated by only one point in F
idx = bsxfun(@and, idx, idx_unique); % the contribution to F(i) is given only if P(j) is dominated uniquely by F(i)
hv = squeeze(sum(idx,1)) / N;