% Generic episodic RL with Importance Sampling.

N = 20;
N_MAX = N * 10;
W = ones(1, N_MAX);
if makeDet, policy = policy.makeDeterministic; end

iter = 1;

% solver = REPSep_Solver(0.9);
% solver = NES_Solver(0.1);
solver = MORE_Solver(0.9,0.99,-75,policy_high);

%% Learning
while true

    [data, avgRew] = collect_episodes(mdp, N, steps_learn, policy_high, policy);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(data.J(robj,:),[],2),1,N_MAX);
        Policies = repmat(policy_high,1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    J = [data.J(robj,:), J(:, 1:N_MAX-N)];
    Theta = [data.Theta, Theta(:, 1:N_MAX-N)];
    Policies = [repmat(policy_high,1,N), Policies(:, 1:N_MAX-N)];
    
    % Compute IS weights
%     W = mixtureIS(policy_high, Policies, N, Theta);

    % Eval current policy
%     [data, avgRew] = collect_episodes(mdp, episodes_eval, steps_eval, policy_high.makeDeterministic, policy);

    % Perform a policy update step
    [policy_high, div] = solver.step(J, Theta, policy_high, W);

    % Store and print info
    J_history(:,iter) = data.J(robj,:);
    fprintf( 'Iter: %d, Avg Reward: %.4f, Div: %.2f, Entropy: %.4f\n', ...
        iter, avgRew(robj), div, policy_high.entropy(data.Theta) );

    iter = iter + 1;
    
end

%%
plothistory(J_history)
show_simulation(mdp, ...
    policy.update(policy_high.makeDeterministic.drawAction), ...
    1000, 0.01)