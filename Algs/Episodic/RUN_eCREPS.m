N = 20;
N_MAX = N * 10;
if makeDet, policy = policy.makeDeterministic; end
bfs_solver = @(varargin)basis_poly(1,mdp.dctx,1,varargin{:});
solver = CREPS_Solver(0.9,bfs_solver);

iter = 1;


%% Run CREPS
while true
    
    [data, avgRew] = collect_episodes(mdp, N, steps_learn, policy_high, policy);
    PhiPolicy_iter = policy_high.basis(data.Context);
    PhiSolver_iter = solver.basis(data.Context);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(data.J,[],2),1,N_MAX);
        random_ctx = mdp.getcontext(N_MAX);
        Theta = policy_high.drawAction(random_ctx);
        PhiPolicy = policy_high.basis(random_ctx);
        PhiSolver = solver.basis(random_ctx);
    end
    
    % Enqueue the new samples and remove the old ones
    J = [data.J, J(1:N_MAX-N)];
    Theta = [data.Theta, Theta(: ,1:N_MAX-N)];
    PhiSolver = [PhiSolver_iter, PhiSolver(:, 1:N_MAX-N)];
    PhiPolicy = [PhiPolicy_iter, PhiPolicy(:, 1:N_MAX-N)];
    
    % Get the weights for policy update
    [weights, divKL] = solver.optimize(J(robj,:), PhiSolver);

    J_history(:,iter) = data.J(robj,:);
    fprintf( '%d) Avg Reward: %.4f, \tKL Div: %.2f, \tEntropy: %.3f\n', ...
        iter, avgRew(robj), divKL, policy_high.entropy );
    
    policy_high = policy_high.weightedMLUpdate(weights, Theta, [ones(1,N_MAX); PhiPolicy]);
    
    iter = iter + 1;

end


%%
plothistory(J_history)