function W = convexWeights ( N, density )
% Generates a matrix W of D weights such that the sum of every row is 1.
%
% Inputs:
% - N       : number of weights.
% - density : number of points in the interval [0,1].
%
% Outputs: 
% - W       : a M-by-N matrix which rows are the weights for convex 
%             combinations between N elements. M depends on density and N.
%
% Example:
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

W = recursiveLoops([], zeros(1,N), 1, N, density);

end

function W = recursiveLoops ( W, w, n, N, density )
% Uses recursion to generate N-1 nested loops.
%
% Inputs:
% - W       : current matrix
% - w       : current weights
% - n       : current level of recursion
% - N       : max level of recursion
% - density : points in the interval [0,1]

if n == N
    v = 0;
    for i = 1 : n - 1
        v = v + w(i);
    end
    w(n) = 1 - v;
    W = [W; w];
    return
end

v = 0;
for i = 1 : n - 1
    v = v + w(i);
end
v = 1 - v;

for i = 0 : 1 / density : v
    w(n) = i;
    W = recursiveLoops(W,w,n+1,N,density);
end

end