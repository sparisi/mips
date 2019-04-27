function p = H_projection(p, beta)
% This function performs a simple projection of a Gaussian policy P to
% ensure that its entropy is equal to BETA.
% In practice, it is a simple rescaling of the covariance matrix.
%
% =========================================================================
% REFERENCE
% R Akrour, J Peters, G Neumann
% Projections for Approximate Policy Iteration Algorithms (2019)

lambda = log(diag(p.U)); % U'*U = Sigma (Choleski decomp)
d = p.daction;
h = 0.5 * d * log(2 * pi * exp(1)) + sum(lambda) - beta;
U = p.U * exp(- h / d);
try
    p = p.update(p.mu, U'*U); % For state-independent policies
catch
    p = p.update(p.A, U'*U); % For state-dependent policies
end
