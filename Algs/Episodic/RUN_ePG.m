N = 20;
N_MAX = N * 10;
if makeDet, policy = policy.makeDeterministic; end

iter = 1;
lrate = 0.1;

%% Learning
while true

    [data_iter, avgRew] = collect_episodes(mdp, N, steps_learn, policy_high, policy);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        data.J = repmat(min(data_iter.J,[],2),1,N_MAX);
        data.Theta = policy_high.drawAction(N_MAX);
    end
    
    % Enqueue the new samples and remove the old ones
    data.J = [data_iter.J, data.J(:, 1:N_MAX-N)];
    data.Theta = [data_iter.Theta, data.Theta(:, 1:N_MAX-N)];
    
    [grad, stepsize] = NESbase(policy_high, data, lrate);
%     [grad, stepsize] = PGPEbase(policy_high, data, lrate);

    J_history(:,iter) = data_iter.J(:,robj);
    fprintf( '%d) Avg Reward: %.4f, \tNorm: %.2f, \tEntropy: %.3f\n', ...
        iter, avgRew(robj), norm(grad(:,robj)), policy_high.entropy(data_iter.Theta) );

    policy_high = policy_high.update(policy_high.theta + grad(:,robj)*stepsize(robj));

    iter = iter + 1;

end

%%
plothistory(J_history)
show_simulation(mdp, policy.update(policy_high.mu), 1000, 0.01)