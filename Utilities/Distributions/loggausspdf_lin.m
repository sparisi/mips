function y = loggausspdf_lin(X, Phi, A, Sigma)
% LOGGAUSSPDF_LIN Logarithm of multivariate normal probability density 
% function with linear mean.
%
% =========================================================================
% ACKNOWLEDGEMENT
% http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model

d = size(X,1);
mu = A * Phi;
X = X - mu;
U = chol(Sigma);

Q = U' \ X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d * log(2*pi) + 2 * sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;

end