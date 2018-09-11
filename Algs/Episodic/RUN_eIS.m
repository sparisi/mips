% Generic episodic RL with Importance Sampling.

N = 20;
N_MAX = N * 10;
W = ones(1, N_MAX);
policy = policy.makeDeterministic; % Learn deterministic low-level policy

iter = 1;

solver = REPSep_Solver(0.9); div_str = 'KL (Weights)';
% solver = NES_Solver(0.1); div_str = 'Grad Norm';
% solver = MORE_Solver(0.9,0.99,-75,policy_high); div_str = 'KL';

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
    awgRew = evaluate_policies(mdp, episodes_eval, steps_eval, policy.update(policy_high.mu));
    J_history(:,iter) = awgRew;

    % Perform a policy update step
    policy_high_old = policy_high;
    [policy_high, div] = solver.step(J, Theta, policy_high, W);

    % Store and print info
    fprintf( ['Iter: %d, Avg Reward: %.4f, ' div_str ': %.2f, Entropy: %.4f'], ...
        iter, avgRew(robj), div, policy_high.entropy(data.Theta) );
    if isa(policy_high,'Gaussian')
        fprintf(', KL: %.4f', kl_mvn(policy_high, policy_high_old));
    end
    fprintf('\n');

    iter = iter + 1;
    
end

%%
show_simulation(mdp, policy.update(policy_high.mu), 1000, 0.01)