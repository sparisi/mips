function d = kl_mvn(p, q, varargin)
% KL_MVN Computes the Kullback–Leibler divergence KL(P||Q) from distribution
% P to distribution Q, both Multivariate Normal.

s0 = p.Sigma;
s1 = q.Sigma;
m0 = p.mu;
m1 = q.mu;
k = length(m0);

d = 0.5 * (trace(s1 \ s0) + (m1 - m0)' / s1 * (m1 - m0) - k + log(det(s1) / det(s0)));
