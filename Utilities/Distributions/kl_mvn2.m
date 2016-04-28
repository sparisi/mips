function d = kl_mvn2(p, q, states)
% KL_MVN2 Computes the Kullback–Leibler divergence KL(P||Q) from distribution
% P to distribution Q, both Multivariate Normal with means linearly
% dependent on the state.

s0 = p.Sigma;
s1 = q.Sigma;
m0 = p.A(:,1);
K0 = p.A(:,2:end);
m1 = q.A(:,1);
K1 = q.A(:,2:end);
dim = length(m0);

const_diff = m0 - m1;
lin_diff = K0 - K1;
mu_states = mean(states,2);
cov_states = cov(states');

d = 0.5 * (trace(s1 \ s0) + const_diff' / s1 * const_diff - dim + logdet(s1,'chol') - logdet(s0,'chol'));

d = d + 0.5 * ( ...
    2 * mu_states' * lin_diff' / s1 * const_diff + ...
    mu_states' * lin_diff' / s1 * lin_diff * mu_states + ...
    trace(lin_diff' / s1 * lin_diff * cov_states) ...
    );
