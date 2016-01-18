function y = loggausspdf(X, mu, Sigma)
% LOGGAUSSPDF Logarithm of multivariate normal probability density function.
%
% =========================================================================
% ACKNOWLEDGEMENT
% http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model

d = size(X,1);
X = bsxfun(@minus, X, mu);
U = chol(Sigma);

Q = U' \ X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d * log(2*pi) + 2 * sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;

end