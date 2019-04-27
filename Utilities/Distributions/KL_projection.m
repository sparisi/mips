function p = KL_projection(p, q, epsilon)
% This function performs a projection of a Gaussian policy P to ensure that 
% KL(p||q) < epsilon.
%
% =========================================================================
% REFERENCE
% R Akrour, J Peters, G Neumann
% Projections for Approximate Policy Iteration Algorithms (2019)

if kl_mvn(p,q) > epsilon
    mq = 0.5 * (q.mu - p.mu)' / q.Sigma * (q.mu - p.mu); % change in mean
    rq = 0.5*(trace(q.Sigma\p.Sigma) - p.daction); % rotation of the covariance
    eq = 0.5*(logdet(q.Sigma,'chol') - logdet(p.Sigma,'chol')); % change in entropy
    eta = epsilon / (mq + rq + eq);
    Sigma = eta*p.Sigma + (1-eta)*q.Sigma;
    p = p.update(p.mu, Sigma);
end

if kl_mvn(p,q) > epsilon
    mq = 0.5 * (q.mu - p.mu)' / q.Sigma * (q.mu - p.mu); % change in mean
    rq = 0.5*(trace(q.Sigma\p.Sigma) - p.daction); % rotation of the covariance
    eq = 0.5*(logdet(q.Sigma,'chol') - logdet(p.Sigma,'chol')); % change in entropy
    eta = sqrt((epsilon - rq - eq) / mq);
    mu = eta*p.mu + (1-eta)*q.mu;
    p = p.update(mu, p.Sigma);
end
