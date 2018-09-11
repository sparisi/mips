% Step-based REPS maximizing Monte-Carlo estimates of the return.

solver = REPSac_Solver(0.1);
nmax = episodes_learn*steps_learn*5;
data = [];
varnames = {'r','s','nexts','a','Q'};
bfsnames = { {'phi', @(s)policy.get_basis(s)} };
iter = 1;

%% Learning
while iter < 1500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    data = getdata(data,ds,nmax,varnames,bfsnames);

    [d, divKL] = solver.optimize(data.Q);
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