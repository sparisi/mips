% Unconstrained REPS. The KL constraint is over pi(a|s) and not over p(s,a).
% Thus it does not learn the V-funciton by minimizing the dual. Instead, 
% Monte Carlo estimates of the expected return are maximized.

solver = REPSep_Solver(0.1);

data = [];
varnames = {'r','s','nexts','a','t'};
bfsnames = { {'phi', @(s)policy.get_basis(s)} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    R = mc_ret(data,mdp.gamma);
    
    [d, divKL] = solver.optimize(R);
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phi);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f,  Eta: %e,  KL (Weights): %.2f,  J: %.4f', ...
        iter, entropy, solver.eta, divKL, J(robj))
    if isa(policy,'Gaussian')
        fprintf(',  KL: %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;

end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 100, 0.01)