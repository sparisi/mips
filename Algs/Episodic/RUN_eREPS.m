N = 20;
N_MAX = N * 10;
solver = REPS_Solver(0.9);
if makeDet, policy = policy.makeDeterministic; end

J = zeros(dreward,N_MAX);
Theta = zeros(policy_high.daction,N_MAX);

iter = 1;

%% Learning
while true
    
    [J_iter, Theta_iter] = collect_episodes(mdp, N, steps_learn, policy_high, policy);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter,[],2),1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
    end

    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(:, 1:N_MAX-N)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    
    [weights, divKL] = solver.optimize(J(robj,:));

    avgRew = mean(J_iter(robj,:));
    J_history(:,iter) = J_iter(robj,:);
    fprintf( '%d) Avg Reward: %.4f, \tKL Div: %.2f, \tEntropy: %.3f\n', ...
        iter, avgRew, divKL, policy_high.entropy );
    
    policy_high = policy_high.weightedMLUpdate(weights, Theta);
    
    iter = iter + 1;

end

%%
plothistory(J_history)
