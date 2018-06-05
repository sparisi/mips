% Actor-critic REPS.
% As REPS_disc, but instead of using Monte-Carlo estimates, it uses GAE to estimate the advantage.

rng(1)

% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
gamma = mdp.gamma;
lambda_trace = 0.95;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

omega = (rand(bfsV(),1)-0.5)*2;
solver = REPSdisc_Solver(0.5);

data = [];
varnames = {'r','s','nexts','a','endsim','Q'};
bfsnames = { {'phiP', @(s)policy.basis_bias(s)}, {'phiV', bfsV} };
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
    A = gae(data,V,gamma,lambda_trace);

    % Update V
    omega = fminunc(@(omega)learn_V(omega,data.phiV,data.Q), omega, options);
    
    % Get REPS weights
    [d, divKL] = solver.optimize(A);
    
    % Print info
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   KL: %.2f,   J: %e \n', ...
        iter, entropy, divKL, J);
    J_history(iter) = J;
    
    % Update pi
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);
    
    iter = iter + 1;
end


%%
function [g, gd] = learn_V(omega, Phi, T)
% Mean squared TD error
V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
end
