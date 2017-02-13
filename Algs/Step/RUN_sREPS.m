% Step-based REPS.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)

solver = sREPS_Solver(1,bfs);

N_MAX = episodes_learn * steps_learn * 2;
Phi = [];
PhiN = [];
Q = [];
Action = [];
iter = 1;

%% Learning
while iter < 500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy(horzcat(ds.s));
    Phi_iter = policy.basis1(horzcat(ds.s));
    PhiN_iter = policy.basis1(horzcat(ds.nexts));
    Action_iter = horzcat(ds.a);
    Q_iter = horzcat(ds.Q);
    Q_iter = Q_iter(robj,:);

    Q = [Q_iter, Q];
    Phi = [Phi_iter, Phi];
    PhiN = [PhiN_iter, PhiN];
    Action = [Action_iter, Action];
    Q = Q(:, 1:min(N_MAX,end));
    Phi = Phi(:, 1:min(N_MAX,end));
    PhiN = PhiN(:, 1:min(N_MAX,end));
    Action = Action(:, 1:min(N_MAX,end));
        
    Q = Q_iter; % Do not re-use previous samples
    Phi = Phi_iter;
    PhiN = PhiN_iter;
    Action = Action_iter;

    [policy, divKL] = solver.step(Q, Action, Phi, PhiN, policy);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f, KL: %.2f, J: %.4f\n', iter, S, divKL, J(robj))
    
    iter = iter + 1;

end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 1000, 0.01)