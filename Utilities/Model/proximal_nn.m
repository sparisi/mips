function X = proximal_nn(X, lambda)
% PROXIMAL_NN Nuclear norm proximal operator.

[U,S,V] = svd(X);
X = U * max(S - lambda * eye(size(S)), 0) * V';