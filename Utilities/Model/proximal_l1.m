function X = proximal_l1(X, lambda)
% PROXIMAL_L1 L1-norm proximal operator.

X = max(abs(X) - lambda, 0) .* sign(X);
