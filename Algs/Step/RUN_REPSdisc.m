% Step-based REPS.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)

solver = REPSdisc_Solver(0.1);
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
    policy = policy.weightedMLUpdate(d, data.a, data.phi);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f, KL: %.2f, J: %.4f\n', iter, entropy, divKL, J(robj))
    
    iter = iter + 1;

end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 100, 0.01)