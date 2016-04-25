function THETA = linear_regression(X, Y, W)
% LINEAR_REGRESSION Weighted linear regression on input-output pairs (X, Y)
% with samples weights W.
%
%    INPUT
%     - X     : [d x N] matrix
%     - Y     : [1 x N] vector
%     - W     : (optional) [1 x N] vector
%
%    OUTPUT
%     - THETA : [d x 1] vector, such that Y = THETA'*X

[D, N] = size(X);
if nargin < 3, W = ones(1,N); end

XW = bsxfun(@times, X, W);
lambda = 1e-3;

if D > N
    A = X'*XW;
    if rank(A) == N, lambda = 0; end
    THETA = XW / (A + lambda*eye(N)) * Y';
    
else
    A = XW*X';
    if rank(A) == D, lambda = 0; end
    THETA = (A + lambda*eye(D)) \ (XW*Y');
end
