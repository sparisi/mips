%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [grad, stepsize] = FiniteDifference(policy, n_samples, J_theta, domain, robj, lrate)

theta = policy.theta;
n_theta = size(theta,1);
var = 1 * eye(n_theta); % change according to the problem
delta_theta = zeros(n_samples,n_theta);
J_perturb = zeros(n_samples,1);
[~, ~, ~, steps] = feval([domain '_settings']);
J_theta_ref = J_theta(robj) * ones(n_samples,1);

parfor i = 1 : n_samples
    theta_perturb = mvnrnd(theta,var)';
    delta_theta(i,:) = theta_perturb - theta;
    tmp_pol = policy;
    tmp_pol.theta = theta_perturb;
    
    [~, J_ep] = collect_samples(domain, 1, steps, tmp_pol);
    J_perturb(i) = J_ep(robj);
end
delta_J = J_perturb - J_theta_ref;

lambda = .9; % Ridge regression factor
grad = ((delta_theta' * delta_theta + lambda * eye(n_theta)) \ delta_theta') * delta_J;

if nargin >= 6
    T = eye(length(grad)); % trasformation in Euclidean space
    lambda = sqrt(grad' * T * grad / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end

end