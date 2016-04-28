function d = kl_mvn(p, q)
% KL_MVN Computes the Kullback–Leibler divergence KL(P||Q) from distribution
% P to distribution Q, both Multivariate Normal with constant mean.

s0 = p.Sigma;
s1 = q.Sigma;
m0 = p.mu;
m1 = q.mu;
dim = length(m0);

d = 0.5 * (trace(s1 \ s0) + (m1 - m0)' / s1 * (m1 - m0) - dim + logdet(s1,'chol') - logdet(s0,'chol'));
