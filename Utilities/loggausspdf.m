function y = loggausspdf(X, mu, Sigma)
% Logarithm of multivariate normal probability density function.
%
% Written by Michael Chen (sth4nth@gmail.com).

d = size(X,1);
X = bsxfun(@minus, X, mu);
[U, p]= chol(Sigma);
if p ~= 0
    error('Sigma is not positive-definite.');
end
Q = U' \ X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d * log(2*pi) + 2 * sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;

end