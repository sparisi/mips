% Unconstrained REPS. Like ACREPS, but it does not constrain on the state 
% distribution and thus it does not learn the value funciton by minimizing 
% the dual. Instead, Monte-Carlo estimates of Q are maximized.

solver = REPSep_Solver(0.1);

data = [];
varnames = {'r','s','nexts','a','endsim'};
bfsnames = { {'phi', @(s)policy.get_basis(s)} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).endsim(end) = 1; % To separate episodes for MC returns
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    Q = mc_ret(data,mdp.gamma);
    
    [d, divKL] = solver.optimize(Q);
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phi);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f,  KL (Weights): %.2f,  J: %.4f', iter, entropy, divKL, J(robj))
    if isa(policy,'Gaussian')
        fprintf(',  KL: %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;

end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 100, 0.01)