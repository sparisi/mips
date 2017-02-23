function [reduce, d] = mykpca(x, kernelType, kernelParam, minExplained)
% MYKPCA Dimensionality Reduction by Kernel Principal Component Analysis.
%
%    INPUT
%     - x            : [M x N] matrix, where N is the number of samples and
%                      M is the dimensionality of each samples
%     - kernelType   : kernel name
%     - kernelParam  : kernel parameter
%     - minExplained : (optional) minimum explained variance (default 1)
%
%    OUTPUT
%     - reduce       : reduction function, i.e., Y = REDUCE(X), where y is 
%                      the matrix of reduced samples of size [d x N]
%     - d            : dimensionality of reduced samples

if nargin < 4, minExplained = 1; end
assert(minExplained <= 1 && minExplained > 0, ...
    'The minimum explained variance must be in (0,1].')

[D, N] = size(x);

% Compute kernel matrix
K0            = kernel(x', x', kernelType, kernelParam);
K0(isnan(K0)) = 0;
K0(isinf(K0)) = 0;
oneN          = ones(N,N) / N;
K             = K0 - oneN*K0 - K0*oneN + oneN*K0*oneN; % Centered kernel matrix

% Compute principal components
[V, L]    = eigs(K,D);
eigVector = real(V);
eigValue  = real(sqrt((L)));
eigVector = eigVector * eigValue;

% Compute explained variance
y = eigVector' * K0;
vP = var(x,[],2);
vP(vP==0) = 1;
vR = var(y,[],2);
vR(vR==0) = 1;
explained = cumsum(vR./vP) / sum(vR./vP);
d = find(explained >= minExplained, 1); % Select reduced dimensionality

eigVector = eigVector(:,1:d);

reduce = @(in) eigVector' * kernel(in', x', kernelType, kernelParam) ;
