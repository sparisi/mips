function THETA = bayesian_linear_regression(X, Y, W)
% BAYESIAN_LINEAR_REGRESSION Weighted Bayesian linear regression on 
% input-output pairs (X, Y) with samples weights W.
% Model: Y = THETA'*X + e, where e ~ N(0, sigma2)
% Prior: THETA ~ N(mu0, S0), where S0 = I * tau2.
%
%    INPUT
%     - X     : [d x N] matrix
%     - Y     : [1 x N] vector
%     - W     : (optional) [1 x N] vector
%
%    OUTPUT
%     - THETA : [d x 1] vector, such that Y = THETA'*X

[d, N] = size(X);
if nargin < 3, W = ones(1,N); end

sigma2 = 100;
tau2 = 100;
invS0 = eye(d) / tau2;
mu0 = zeros(d,1);

invS = (1 / sigma2) * (bsxfun(@times, X, W) * X') + invS0;
THETA = invS \ ...
    ( (invS0 * mu0) + (1 / sigma2) * bsxfun(@times, X, W) * Y' );
