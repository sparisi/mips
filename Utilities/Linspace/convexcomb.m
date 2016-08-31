function M = convexcomb(D, U)
% CONVEXCOMB Generates a matrix M of convex combinations of D-dimensional 
% points in the interval [0,1].
%
%    INPUT
%     - D : dimension of the points
%     - U : number of unique points in the interval [0,1] for each
%           combination
%
%    OUTPUT
%     - M : [N x D] matrix, where rows are D-dimensional convex  
%           combinations of. N is the binomial coefficient of (U+D-1, D-1)
%
% =========================================================================
% EXAMPLE
% convexcomb(2,10)
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

c = nchoosek(1 : U+D-1, D-1);
m = size(c,1);
t = ones(m, U+D-1);
t( repmat((1:m).', 1, D-1)+(c-1)*m ) = 0;
u = [zeros(1,m); t.'; zeros(1,m)];
v = cumsum(u,1);
M = diff( reshape(v(u==0), D+1, m), 1 ).';
M = M ./ U;
