function X = mymvnrnd(mu, sigma, n)
% MYMVNRND Draws N samples from a multivariate normal distribution of mean
% MU and covariance SIGMA. MU can be either a row or a column, but the
% output X will always be a [D x N] matrix, where D is the dimension of the
% multivariate.

if ~iscolumn(mu), mu = mu'; end
if nargin == 2, n = 1; end

U = chol(sigma);
X = U'*randn(length(mu),n) + mu;
