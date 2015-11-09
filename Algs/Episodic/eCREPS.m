%% Init domain and low-level policy
clear all

domain = 'damC';
robj = 1;
context_fun = [domain '_context'];
dim_ctx = length(feval(context_fun));
[~, pol_low, ~, steps] = feval([domain '_settings']);

% If the low-level policy has a learnable variance, we don't want to learn 
% it and we make it deterministic
dim_theta = size(pol_low.theta,1) - pol_low.dim_explore;
pol_low = pol_low.makeDeterministic;


%% Init CREPS and high-level policy
phi_policy = @(varargin)basis_poly(1,dim_ctx,1,varargin{:});
phi_vfun = @(varargin)basis_poly(1,dim_ctx,1,varargin{:});

K0 = zeros(dim_theta,phi_policy());
mu0 = zeros(dim_theta,1);
Sigma0 = 100 * eye(dim_theta); % change according to the domain

% pol_high = gaussian_linear(phi_policy, dim_theta, K0, Sigma0);
% pol_high = gaussian_diag_linear(phi_policy, dim_theta, K0, Sigma0);
pol_high = gaussian_linear_full(phi_policy, dim_theta, mu0, K0, Sigma0);

epsilon = 0.9;
N = 50; % number of rollouts per iteration
N_MAX = N*10; % max number of rollouts used for the policy update
MAX_ITER = 100;
solver = CREPS_Solver(epsilon,pol_high,phi_vfun);

J = zeros(1,N_MAX);
PhiVfun = zeros(N_MAX,solver.basis());
PhiPolicy = zeros(N_MAX,solver.policy.basis());
Theta = zeros(N_MAX,dim_theta);

J_history = [];
iter = 0;


%% Run CREPS
while iter < MAX_ITER
    
    iter = iter + 1;
    
    [J_iter, Theta_iter, PhiPolicy_iter, PhiVfun_iter] = ...
        collect_episodes_ctx(domain, N, solver);
    
    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter,[],2),1,N_MAX);
        parfor k = 1 : N_MAX
            random_context = feval(context_fun); % generate random context
            PhiPolicy(k,:) = solver.policy.basis(random_context);
            PhiVfun(k,:) = solver.basis(random_context);
            Theta(k,:) = solver.policy.drawAction(random_context);
        end
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(1:N_MAX-N)];
    Theta = [Theta_iter; Theta(1:N_MAX-N, :)];
    PhiVfun = [PhiVfun_iter; PhiVfun(1:N_MAX-N, :)];
    PhiPolicy = [PhiPolicy_iter; PhiPolicy(1:N_MAX-N, :)];
    
    % Get the weights for policy update
    [weights, divKL] = solver.optimize(J(robj,:), PhiVfun);

    avgRew = mean(J_iter(robj,:));
    J_history = [J_history, J_iter'];
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f, Entropy: %.3f\n', ...
        iter, avgRew, divKL, solver.policy.entropy );
    
    % Stopping condition
    if divKL < 1e-3
        break
    else
        solver.update(weights, Theta, PhiPolicy);
    end

end

plothistory(J_history)