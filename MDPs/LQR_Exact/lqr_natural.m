function [ngrad_mu, ngrad_sigma] = lqr_natural(A, B, Q, R, K, Sigma, g)

P = lqr_pmatrix(A,B,Q,R,K,g);

ngrad_mu = 2 * K' * (R + g * B' * P *B) * Sigma + ...
    2 * g * A' * P * B * Sigma;

ngrad_sigma = 2 * Sigma * (R + g * B' * P * B) * Sigma;

end