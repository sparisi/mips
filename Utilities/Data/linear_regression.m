function THETA = linear_regression(X, Y, varargin)
% LINEAR_REGRESSION Weighted linear regression on input-output pairs (X, Y).
%
%    INPUT
%     - X      : [d x N] matrix
%     - Y      : [1 x N] vector
%     - W      : (optional) [1 x N] vector
%     - lambda : (optional) regularization coefficient
%
%    OUTPUT
%     - THETA  : [d x 1] vector, such that Y = THETA'*X

[D, N] = size(X);

options = {'weights', 'lambda'};
defaults = {ones(1,N), NaN};
[W, lambda] = internal.stats.parseArgs(options, defaults, varargin{:});

XW = bsxfun(@times, X, W);

if D > N
    A = X'*XW;
    if isnan(lambda) % If lambda is not specified, use pinv
        THETA = XW * ( pinv(A) * Y' );
    else
        THETA = XW * ( (A + lambda*eye(N)) \ Y' );
    end
    
else
    A = XW*X';
    if isnan(lambda) % If lambda is not specified, use pinv
        THETA = pinv(A) * (XW * Y');
    else
        THETA = (A + lambda*eye(D)) \ (XW * Y');
    end
end
