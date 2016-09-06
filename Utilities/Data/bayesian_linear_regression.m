function THETA = bayesian_linear_regression(X, Y, varargin)
% BAYESIAN_LINEAR_REGRESSION Weighted Bayesian linear regression on 
% input-output pairs (X, Y).
% Model: Y = THETA'*X + e, where e ~ N(0, sigma2)
% Prior: THETA ~ N(mu0, S0), where S0 = I * tau2.
%
%    INPUT
%     - X          : [d x N] matrix
%     - Y          : [1 x N] vector
%     - W          : (optional) [1 x N] vector
%     - noise      : (optional) prior input noise
%     - model_mean : (optional) prior model mean
%     - model_cov  : (optional) prior model covariance
%
%    OUTPUT
%     - THETA      : [d x 1] vector, such that Y = THETA'*X

[d, N] = size(X);

options = {'weights', 'noise', 'model_mean', 'model_cov'};
defaults = {ones(1,N), 100, zeros(d,1), 100};
[W, sigma2, mu0, tau2] = internal.stats.parseArgs(options, defaults, varargin{:});

invS0 = eye(d) / tau2;

invS = (1 / sigma2) * (bsxfun(@times, X, W) * X') + invS0;
THETA = invS \ ...
    ( (invS0 * mu0) + (1 / sigma2) * bsxfun(@times, X, W) * Y' );
