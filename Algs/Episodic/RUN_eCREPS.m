N = 20;
N_MAX = N * 10;
bfs_solver = @(varargin)basis_poly(1,mdp.dctx,1,varargin{:});
solver = CREPS_Solver(0.9,bfs_solver);

J = zeros(dreward,N_MAX);
Theta = zeros(policy_high.daction,N_MAX);

PhiSolver = zeros(N_MAX,solver.basis());
PhiPolicy = zeros(N_MAX,policy_high.basis());

iter = 1;


%% Run CREPS
while true
    
    [J_iter, Theta_iter, Context] = collect_episodes(mdp, N, steps_learn, policy_high, policy);
    PhiPolicy_iter = policy_high.basis(Context);
    PhiSolver_iter = solver.basis(Context);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter,[],2),1,N_MAX);
        random_ctx = mdp.getcontext(N_MAX);
        Theta = policy_high.drawAction(random_ctx);
        PhiPolicy = policy_high.basis(random_ctx);
        PhiSolver = solver.basis(random_ctx);
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(1:N_MAX-N)];
    Theta = [Theta_iter, Theta(: ,1:N_MAX-N)];
    PhiSolver = [PhiSolver_iter, PhiSolver(:, 1:N_MAX-N)];
    PhiPolicy = [PhiPolicy_iter, PhiPolicy(:, 1:N_MAX-N)];
    
    % Get the weights for policy update
    [weights, divKL] = solver.optimize(J(robj,:), PhiSolver);

    avgRew = mean(J_iter(robj,:));
    J_history(:,iter) = J_iter(robj,:);
    fprintf( '%d) Avg Reward: %.4f, \tKL Div: %.2f, \tEntropy: %.3f\n', ...
        iter, avgRew, divKL, policy_high.entropy );
    
    policy_high = policy_high.weightedMLUpdate(weights, Theta, PhiPolicy);
    
    iter = iter + 1;

end


%%
plothistory(J_history)