function [Xn, Xmu, Xstd] = standardize_data(X, W)
% STANDARDIZE_DATA Performs data standardization with importance sampling 
% weights. Resulting data has 0 mean and 1 std.
%
%    INPUT
%     - X    : input data, [d x N] matrix
%     - W    : (optional) importance sampling weights, [1 x N] vector (1 by
%              default)
%
%    OUTPUT
%     - Xn   : standardized data, [d x N] matrix 
%     - Xmu  : data mean, [d x 1] vector
%     - Xstd : data std, [d x 1] vector

if nargin < 2, W = ones(1, size(X,2)); end

W = W / sum(W);
Xmu = sum(bsxfun(@times,X,W),2);

Xstd = sqrt( ...
    sum( ...
    bsxfun( @times, ...
    bsxfun( @minus, X, Xmu ).^2, W ), 2 ) );
Xstd(Xstd<1e-6) = 1e-6; % Regularizer

Xn = bsxfun(@minus, X, Xmu);
Xn = bsxfun(@times, Xn, 1 ./ Xstd);
