function J = evaluate_policies ( policies, domain, makeDet )
% Given a set of low-level POLICIES, it returns the corresponding return J
% in the objectives space.
% Set MAKEDET to 1 if you want to make the policies deterministic.

[n_obj, ~, episodes, steps] = feval([domain '_settings']);
mdp_vars = feval([domain '_mdpvariables']);
isStochastic = mdp_vars.isStochastic;
if makeDet && ~isStochastic
    episodes = 1;
end

N_pol = numel(policies);
J = zeros(N_pol, n_obj);

parfor i = 1 : N_pol
    
%     fprintf('Evaluating policy %d of %d ...\n', i, N_pol)
    
    if makeDet
        policy = policies(i).makeDeterministic;
    else
        policy = policies(i);
    end
    
%     [~, J_sample] = collect_samples(domain, episodes, steps, policy);
    [~, J_sample] = collect_samples_rele(domain, episodes, steps, policy);
    
    J(i,:) = J_sample;

end

end
