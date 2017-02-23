function [reduce, d, reconstruct, nmse] = mypca(x, minExplained)
% MYPCA Dimensionality Reduction by Principal Component Analysis. PCA is 
% performed on the covariance matrix and the data is centered. The desired 
% dimensionality is computed by thresholding the minimum variance explained 
% by the principal components.
%
%    INPUT
%     - x            : [M x N] matrix, where N is the number of samples and
%                     M is the dimensionality of each samples
%     - minExplained : (optional) minimum explained variance (default 1)
%
%    OUTPUT
%     - reduce       : reduction function, i.e., Y = REDUCE(X), where y is 
%                      the matrix of reduced samples of size [d x N]
%     - d            : dimensionality of reduced samples
%     - reconstruct  : reconstruction function, i.e., X = RECONSTRUCT(Y)
%     - nmse         : normalized mean squared error between X and
%                      RECONSTRUCT(REDUCE(X))

if nargin == 1, minExplained = 1; end
assert(minExplained <= 1 && minExplained > 0, ...
    'The minimum explained variance must be in (0,1].')

[~, eigva, eigve] = svd(cov(x')); % SVD already sorts the eigenvalues
eigva = diag(eigva);

explained = cumsum(eigva) / sum(eigva);
d = find(explained >= minExplained, 1); % Select reduced dimensionality
T = eigve(:,1:d); % Transformation matrix

reduce = @(in) T' * bsxfun( @minus, in, mean(x,2) );
reconstruct = @(in) bsxfun( @plus, T * in, mean(x,2) );

if nargout == 4
    nmse = mean( mean( (x - reconstruct(reduce(x))).^2 ) ) / ...
        mean( mean( x.^2 ) );
end
