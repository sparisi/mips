function M = convexcomb(N, U)
% CONVEXCOMB Generates a matrix M of convex combinations of N points in the
% interval [0,1].
%
%    INPUT
%     - N : number of points
%     - U : number of unique points in the interval [0,1] for each
%           combination
%
%    OUTPUT
%     - M : [M x N] matrix which rows are the weights for convex 
%           combinations between N elements. M depends on U and N
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

c = nchoosek(1 : U+N-1, N-1);
m = size(c,1);
t = ones(m, U+N-1);
t( repmat((1:m).', 1, N-1)+(c-1)*m ) = 0;
u = [zeros(1,m); t.'; zeros(1,m)];
v = cumsum(u,1);
M = diff( reshape(v(u==0), N+1, m), 1 ).';
M = M ./ U;

end