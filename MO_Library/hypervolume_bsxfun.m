function h = hypervolume_bsxfun(F, AU, U, N)
% Same as HYPERVOLUME, but it uses BSXFUN. Its speed depends on the size of
% F and on N.

dim = length(U);
P = bsxfun( @plus, AU, bsxfun(@times, (U - AU), rand(N, dim)) ); % hypercuboid
tmp = permute(F, [3,2,1]);

idx = sum(bsxfun(@le, P, tmp), 2) == dim; % (i,j) says is point P(i) is dominated by F(j)
h = sum(idx,3); % says by how many points in F a point in P is dominated
h = sum(h>0) / N; % count each point in P only once