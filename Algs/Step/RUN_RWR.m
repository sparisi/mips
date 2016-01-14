% Reward-Weighted Regression.
%
% =========================================================================
% REFERENCE
% J Peters, S Schaal
% Reinforcement Learning by Reward-weighted Regression for Operational 
% Space Control (2007)

N_MAX = 10000;
Phi = [];
R = [];
Action = [];
iter = 1;

%% Learning
while true
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy(horzcat(ds.s));
    Phi_iter = policy.basis1(horzcat(ds.s));
    Action_iter = horzcat(ds.a);
    R_iter = horzcat(ds.Q);

    R = [R_iter, R];
    Phi = [Phi_iter, Phi];
    Action = [Action_iter, Action];
    R = R(:, 1:min(N_MAX,end));
    Phi = Phi(:, 1:min(N_MAX,end));
    Action = Action(:, 1:min(N_MAX,end));

    weights = (R(robj,:) - min(R(robj,:),[],2)) / (max(R(robj,:),[],2) - min(R(robj,:),[],2)); % simple normalization in [0,1]
%     beta = 0.1; weights = exp(beta*R(robj,:));
    
    [~, J] = collect_samples(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d ) Entropy: %.2f, J: %.4f\n', iter, S, J(robj))
    
    policy = policy.weightedMLUpdate(weights, Action, Phi);
    
    iter = iter + 1;

end