function [Xn, Xmu, Xstd] = normalize_data(X, W)
% NORMALIZE_DATA Performs data normalization with importance sampling 
% weights. Resulting data has 0 mean and 1 std.
%
%    INPUT
%     - X    : input data, [d x N] matrix
%     - W    : (optional) importance sampling weights, [1 x N] vector (1 by
%              default)
%
%    OUTPUT
%     - Xn   : normalized data, [d x N] matrix 
%     - Xmu  : data mean, [d x 1] vector
%     - Xstd : data std, [d x 1] vector

if nargin < 2, W = ones(1, size(X,2)); end % IS weights

W = W / sum(W);
Xmu = sum(bsxfun(@times,X,W),2);

Xstd = sqrt( ...
    sum( ...
    bsxfun( @times, ...
    bsxfun( @minus, X, Xmu ).^2, W ), 2 ) );

Xn = bsxfun(@minus, X, Xmu);
Xn = bsxfun(@times, Xn, 1 ./ Xstd);
