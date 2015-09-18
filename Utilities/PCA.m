function [y, eigVector, eigValue] = PCA(x, d, w)
% PCA Principal Component Analysis.
%
%    INPUT
%     - x         : N-by-M matrix, where N is the number of samples and M
%                   the dimension of each samples
%     - d         : (optional) desired dimension of the output (d <= M)
%     - w         : (optional) N-by-1 vector of observation weights
%
%    OUPUT
%     - y         : reduced data (first d principal components)
%     - eigVector : eigenvectors of the covariance matrix C = x'x
%     - eigValue  : eigenvalues of the covariance matrix C = x'x

if nargin < 3
    w = ones(size(x,1),1);
end
if nargin < 2
    d = size(x,2);
end

% Normalize weights
w              = w / sum(w);

% Weight observations
x              = bsxfun(@times,x,sqrt(w));

% Get eigenvalues and eigenvectors
[V, D]         = eigs(cov(x),d);
V              = real(V);
eigValue       = diag(real(D));
[eigValue,idx] = sort(eigValue, 'descend');
eigVector      = V(:,idx);

% Normalization
eigVector      = normc(eigVector);

% Dimensionality reduction
eigVectorR     = eigVector(:,1:d);
y              = x * eigVectorR;
