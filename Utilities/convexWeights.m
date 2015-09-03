function w = convexWeights(dim, n)
% CONVEXWEIGHTS Generates a matrix W of convex points in [0,1].
%
%    INPUT
%     - dim : number of weights
%     - n   : number of unique points in the interval [0,1] for each weight
%
%    OUTPUT
%     - w   : M-by-N matrix which rows are the weights for convex 
%             combinations between N elements. M depends on density and N
%
% =========================================================================
% EXAMPLE
% convexWeights(2,10)
% [0.0 1.0
%  0.1 0.9
%  0.2 0.8
%  0.3 0.7
%  0.4 0.6
%  0.5 0.5
%  0.6 0.4
%  0.7 0.3
%  0.8 0.2
%  0.9 0.1
%  1.0 0.0]

c = nchoosek(1 : n+dim-1, dim-1);
m = size(c,1);
t = ones(m, n+dim-1);
t( repmat((1:m).', 1, dim-1)+(c-1)*m ) = 0;
u = [zeros(1,m); t.'; zeros(1,m)];
v = cumsum(u,1);
w = diff( reshape(v(u==0), dim+1, m), 1 ).';
w = w ./ n;

end