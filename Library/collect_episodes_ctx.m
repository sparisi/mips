function [J, Theta, PhiPolicy, PhiVfun] = ...
    collect_episodes_ctx(domain, maxepisodes, solver)
% COLLECT_EPISODES_CTX Collects contextual episodes for the specified 
% domain. The low-level policy is deterministic and its parameters are 
% drawn from a high-level distribution (SOLVER.POLICY).

[n_obj, pol_low, ~, steps] = feval([domain '_settings']);
context_fun = [domain '_context'];
pol_low = pol_low.makeDeterministic;

dim_theta = solver.policy.dim;

PhiPolicy = zeros(maxepisodes,solver.policy.basis());
PhiVfun = zeros(maxepisodes,solver.basis());
Theta = zeros(maxepisodes,dim_theta);
J = zeros(n_obj,maxepisodes);

parfor k = 1 : maxepisodes

    % Get context
    context = feval(context_fun);
    PhiPolicy(k,:) = solver.policy.basis(context);
    PhiVfun(k,:) = solver.basis(context);
    
    % Draw theta from the high-level distribution
    theta = solver.policy.drawAction(context);
    pol_tmp = pol_low;
    pol_tmp.theta(1:dim_theta) = theta; % set only the mean, not the variance
    Theta(k,:) = theta;
    
    % Rollout
    [~, J_ep] = collect_samples_ctx(domain, 1, steps, pol_tmp, context);
    J(:,k) = J_ep;

end

end
