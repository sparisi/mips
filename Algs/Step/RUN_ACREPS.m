% Actor-critic REPS.
% As REPS_disc, but instead of using Monte-Carlo estimates, it uses GAE to estimate the advantage.

rng(1)

% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'Hessian', 'on', ...
    'MaxFunEvals', 100, ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
lambda_trace = 0.95;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

omega = (rand(bfsV(),1)-0.5)*2;
solver = REPSac_Solver(0.3);

data = [];
varnames = {'r','s','nexts','a','endsim','Q'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1000
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).endsim(end) = 1;
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    
    % Estimate A
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace);

    % Update V
    omega = fminunc(@(omega)learn_V(omega,data.phiV,data.Q), omega, options);
    
    % Get REPS weights
    [d, divKL] = solver.optimize(A);
    
    % Print info
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    
    % Update pi
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);
    
    fprintf('%d) Entropy: %.4f,  KL (Weights): %.4f,  J: %e', ...
        iter, entropy, divKL, J);
    if isa(policy,'Gaussian')
        fprintf(',  KL: %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;
end


%%
function [g, gd, h] = learn_V(omega, Phi, T)
% Mean squared TD error
V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
h = Phi*Phi'/size(T,2);
end
