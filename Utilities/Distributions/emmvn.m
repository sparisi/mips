function [model, llh] = emmvn(X, W)
% EMMVN Weighted EM for fitting a multivariate Gaussian distribution.
%
%    INPUT
%     - X    : [D x N] data matrix
%     - W    : (optional) [N x 1] weights vector (1 by default)
%
%    OUTPUT
%     - model: struct with the mean (mu) and the covariance (Sigma) of the
%              Gaussian distribution
%     - llh  : log-likelihood of the model

if nargin == 1
    W = ones(size(X,2),1);
else
    assert(min(W>=0), 'Weights cannot be negative.');
end

tol = 1e-10;
maxiter = 500;
llh = -inf(1, maxiter);
converged = false;
t = 1;

while ~converged && t < maxiter
    t = t + 1;
    model = maximization(X, W);
    llh(t) = expectation(X, model);
    converged = llh(t) - llh(t-1) < tol * abs(llh(t));
end
llh = llh(2:t);

if ~converged
    warning('Not converged in %d steps.\n',maxiter);
end

end


%% Expectation
function llh = expectation(X, model)

mu = model.mu;
Sigma = model.Sigma;

logRho = loggausspdf(X, mu, Sigma);
T = logsumexp(logRho,2);
llh = sum(T); % loglikelihood

end


%% Maximization
function model = maximization(X, W)

mu = X * W / sum(W);
Xo = bsxfun(@minus, X, mu);
Xw = bsxfun(@times, Xo, sqrt(W'));
Sigma = Xw * Xw';
Z = (sum(W)^2 - sum(W.^2)) / sum(W);
Sigma = Sigma / Z;
Sigma = nearestSPD(Sigma);

model.mu = mu;
model.Sigma = Sigma;

end
