% Generic episodic RL with Importance Sampling.

N = 20;
N_MAX = N * 10;
if makeDet, policy = policy.makeDeterministic; end

iter = 1;

solver = REPS_Solver(0.9);
% solver = NES_Solver(0.1);

%% Learning
while true

    [J_iter, Theta_iter] = collect_episodes(mdp, N, steps_learn, policy_high, policy);
    Policies_iter = repmat(policy_high,1,N);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter(robj,:),[],2),1,N_MAX);
        Policies = repmat(policy_high,1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    J = [J_iter(robj,:), J(:, 1:N_MAX-N)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    Policies = [Policies_iter, Policies(:, 1:N_MAX-N)];
    
    % Compute IS weights
    W = mixtureIS(policy_high, Policies, Theta, N);

    % Perform an update step
    [policy_high, div] = solver.step(J, Theta, policy_high, W);

    avgRew = mean(J_iter(robj,:));
    J_history(:,iter) = J_iter(robj,:);
    fprintf( 'Iter: %d, Avg Reward: %.4f, Div: %.2f\n', ...
        iter, avgRew, div );

    iter = iter + 1;
    
end

%%
plothistory(J_history)
