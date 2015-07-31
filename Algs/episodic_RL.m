clear all
domain = 'deep';
robj = 1;

[n_obj, pol_high] = settings_episodic(domain,1);

N = 40;
N_MAX = N*10;

% solver = REPS_Solver(0.9,N_MAX,pol_high);
solver = NES_Solver(1,N_MAX,pol_high);

J = zeros(N_MAX,n_obj);
Theta = zeros(pol_high.dim,N_MAX);

J_history = [];
iter = 0;

%% Learning
while true

    iter = iter + 1;
    
    [J_iter, Theta_iter] = collect_episodes(domain, N, solver.policy);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = solver.policy.drawAction;
        end
    end
        
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];

    % Perform an update step
    div = solver.step(J(:,robj), Theta);
    
    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_iter(:,robj)];
    fprintf( 'Iter: %d, Avg Reward: %.4f, Div: %.2f\n', ...
        iter, avgRew, div );

    % Ending condition
    if div < 0.005
        break
    end
    
end

%% Plot results
figure; shadedErrorBar(1:size(J_history,2), ...
    mean(J_history), ...
    2*sqrt(diag(cov(J_history))), ...
    {'LineWidth', 2'}, 1);
xlabel('Iterations')
ylabel('Average return')