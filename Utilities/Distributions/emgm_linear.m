function [label, model, llh] = emgm_linear(X, Phi, init, W)
% EMGM_LINEAR EM Weighted EM for fitting a Gaussian mixture model with 
% linear means.
%
%    INPUT
%     - X     : [D x N] data matrix
%     - Phi   : [M x N] features matrix
%     - init  : scalar K or 
%               [1 x N] matrix of labels, where 1 <= label(i) <= K, or 
%               model (struct)
%     - W     : (optional) [1 x N] weights vector (1 by default)
%
%    OUTPUT
%     - label : [1 x N] vector indicating which Gaussian is associated to
%               each sample
%     - model : struct with means (mu), covariances (Sigma) and component
%               proportions of the mixture model
%     - llh   : log-likelihood of the model
%
% =========================================================================
% ACKNOWLEDGEMENT
% http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model

if nargin < 4
    W = ones(1,size(X,2));
else
    assert(min(W>=0), 'Weights cannot be negative.');
end

R = initialization(X, Phi, init);
[~, label(1,:)] = max(R, [], 2);
R = R(:, unique(label));

tol = 1e-10;
maxiter = 500;
llh = -inf(1, maxiter);
converged = false;
t = 1;

while ~converged && t < maxiter

    t = t + 1;
    model = maximization(X, Phi, R, W);
    [R, llh(t)] = expectation(X, Phi, model);
    
    [~, label(:)] = max(R, [], 2);
    u = unique(label);   % non-empty components
    if size(R, 2) ~= size(u, 2)
        R = R(:, u);     % remove empty components
    else
        converged = llh(t) - llh(t-1) < tol * abs(llh(t));
    end
    
end

llh = llh(2:t);
if ~converged
    warning('Not converged in %d steps.\n',maxiter);
end

end


%% Init
function R = initialization(X, Phi, init)

[d,n] = size(X);
if isstruct(init)         % initialize with a model
    R = expectation(X, Phi, init);
elseif length(init) == 1  % random initialization with k random components
    A = zeros(d,size(Phi,1),init);
    for k = 1 : init
        idx = randsample(n,d);
        A(:,:,k) = ( pinv(Phi(:,idx))' * X(:,idx) )';
    end
    model.A = A;
    model.Sigma = repmat(nearestSPD(cov(X')),1,1,k);
    model.ComponentProportion = ones(1,init)/init;
    R = expectation(X, Phi, model);
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('Initialization is not valid.');
end

end

%% Expectation
function [R, llh] = expectation(X, Phi, model)

A = model.A;
Sigma = model.Sigma;
w = model.ComponentProportion;

n = size(X,2);
k = length(model.ComponentProportion);
logRho = zeros(n,k);

for i = 1 : k
    logRho(:,i) = loggausspdf_lin(X, Phi, A(:,:,i), Sigma(:,:,i));
end

logRho = bsxfun(@plus, logRho, log(w));
T = logsumexp(logRho,2);
llh = sum(T) / n; % loglikelihood
logR = bsxfun(@minus, logRho, T);
R = exp(logR);

end


%% Maximization
function model = maximization(X, Phi, R, W)

d = size(X,1);
f = size(Phi,1);
k = size(R,2);
R = bsxfun(@times,R,W');
nk = sum(R,1)';
w = nk / sum(W);

Sigma = zeros(d,d,k);
A = zeros(d,f,k);
sqrtR = sqrt(R);
for i = 1 : k
    Z = (sum(R(:,i))^2 - sum(R(:,i).^2)) / sum(R(:,i))^2;
    PhiD = bsxfun(@times,Phi,R(:,i)');
    tmp = PhiD * Phi';
    if rank(tmp) == f
        A(:,:,i) = (tmp \ PhiD * X')';
    else
        A(:,:,i) = (pinv(tmp) * PhiD * X')';
    end
    mu = A(:,:,i) * Phi;
    Xo = bsxfun(@minus, X, mu);
    Xo = bsxfun(@times, Xo, sqrtR(:,i)');
    Sigma(:,:,i) = Xo * Xo' / nk(i);
    if Z ~= 0, Sigma(:,:,i) = Sigma(:,:,i) / Z; end
    Sigma(:,:,i) = nearestSPD(Sigma(:,:,i));
end

model.A = A;
model.Sigma = Sigma;
model.ComponentProportion = w';

end
