function [model, nmse] = quadraticfit(X, Y, varargin)
% QUADRATICFIT Learns a quadratic model to fit input-output pairs (X, Y):
% Y = X'*R*X + X'*r + r0 (R is symmetric and negative definite). 
% For the likelihood of the linear model, a multiplicative noise model is 
% used (the higher the absolute value of Y, the higher the variance).
%
%    INPUT
%     - X           : [d x N] matrix, where N is the number of samples
%     - Y           : [1 x N] vector
%     - weights     : (optional) [1 x N] vector of samples weights
%     - standardize : (optional) flag to standardize X and Y
%
%    OUTPUT
%     - model       : struct with fields R, r, r0, dim (number of 
%                     parameters of the model) and eval (function for 
%                     generating samples, i.e., Y = model.eval(X))
%     - nmse        : normalized mean squared error on the training data

[d, N] = size(X);

options = {'weights', 'standardize'};
defaults = {ones(1,N), 0};
[W, standardize] = internal.stats.parseArgs(options, defaults, varargin{:});

D = d*(d+1)/2 + d + 1; % Number of parameters of the quadratic model

if standardize
    [Xn, Xmu, Xstd] = standardize_data(X, W);
    [Yn, Ymu, Ystd] = standardize_data(Y, W);
else
    Xn = X;
    Yn = Y;
    Xmu = zeros(d,1);
    Xstd = ones(d,1);
    Ymu = zeros(1,1);
    Ystd = ones(1,1);
end

% Generate features vector for linear regression
Phi = basis_quadratic(d,Xn);

% Get parameters by linear regression on pairs (Phi, Y)
params = linear_regression(Phi, Yn, 'weights', W);
assert(~any(isnan(params)), 'Model fitting failed.')

% Extract model params
AQuadratic = zeros(d);
ind = logical(tril(ones(d)));
AQuadratic(ind) = params(d+2:end);
R = (AQuadratic + AQuadratic') / 2;
% r = params(2:d+1);
% r0 = params(1);

% Enforce R to be negative definite
[U, V] = eig(R);
V(V > 0) = -1e-8;
R = U * V * U';

% Re-learn r and r0
quadYn = sum( (Xn'*R)' .* Xn, 1 );
linearPhi = basis_poly(1, d, 1, Xn);
params = linear_regression(linearPhi, Yn - quadYn, 'weights', W);
r0 = params(1);
r = params(2:end);

% De-standardize
XstdMat = diag((1./Xstd));
YstdMat = diag(Ystd);
A = (XstdMat * R * XstdMat);
newR = YstdMat * A;
newr = -YstdMat * (2*A*Xmu - XstdMat*r);
newr0 = YstdMat * (Xmu'*A*Xmu - r'*XstdMat*Xmu + r0) + Ymu;
R = newR;
r = newr;
r0 = newr0;

model.dim = D;
model.R = R;
model.r = r;
model.r0 = r0;
model.eval = @(X) sum( (X'*R)' .* X, 1 ) + (X'*r)' + r0';

if nargout > 1
    nmse = mean( ( Y - model.eval(X) ).^2 ) / mean( Y.^2 );
end