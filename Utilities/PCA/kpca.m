function [y, eigVector, eigValue, explained] = kpca(x, d, type, varargin)
% KPCA Kernel Principal Component Analysis.
%
%    INPUT
%     - x         : [N x M], where N is the number of samples and M is
%                   the dimensionality of each samples
%     - d         : desired dimensionality of the output (d <= M)
%     - type      : kernel type
%     - varargin  : parameters of the kernel
%
%    OUTPUT
%     - y         : reduced data (first d principal components)
%     - eigVector : eigenvectors of the kernel matrix
%     - eigValue  : rooted eigenvalues of the kernel matrix
%     - explained : variance explained by each component

[N, D] = size(x);

% Compute kernel matrix
K0 = kernel(x, x, type, varargin{:}); 
K0(isnan(K0) | isinf(K0)) = 0;

% Center kernel matrix
oneN      = ones(N,N) / N;
K         = K0 - oneN*K0 - K0*oneN + oneN*K0*oneN; 

% Get eigenvalues and eigenvectors
[V, L]    = eigs(K,D);
eigVector = real(V);
eigValue  = real(sqrt((L)));
eigVector = eigVector * eigValue;
y         = K0 * eigVector; % Project kernel matrix into principal component space

% Compute explained variance
vP        = var(x);
vR        = var(y);
explained = cumsum(vR./vP) / sum(vR./vP);

% Return only desired dimensions
y         = K0 * eigVector(:,1:d);
