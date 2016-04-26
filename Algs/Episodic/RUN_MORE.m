N = 20;
N_MAX = N * 10;
if makeDet, policy = policy.makeDeterministic; end

iter = 1;

solver = MORE_Solver(0.9,0.99,-75,policy_high);

%% Learning
while true

    [ds, avgRew] = collect_episodes(mdp, N, steps_learn, policy_high, policy);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(ds.J(robj,:),[],2),1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    J = [ds.J(robj,:), J(:, 1:N_MAX-N)];
    Theta = [ds.Theta, Theta(:, 1:N_MAX-N)];

    % Eval current policy
%     [ds, avgRew] = collect_episodes(mdp, episodes_eval, steps_eval, policy_high.makeDeterministic, policy);
    J_history(:,iter) = ds.J(robj,:);

    % Perform an update step
    [policy_high, divKL] = solver.step(J,Theta,policy_high);
    
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f, Entropy: %.4f \n', ...
        iter, avgRew(robj), divKL, policy_high.entropy);
    
    iter = iter + 1;
    
end

%%
plothistory(J_history)
