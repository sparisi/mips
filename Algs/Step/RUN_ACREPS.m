% In the original paper by Wirth et al., Q is learned by LSTD. However,
% using Monte-Carlo samples (as in this script) works just fine.

mdp.gamma = 0.99;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

solver = ACREPS_Solver(0.3,bfsV);

data = [];
varnames = {'r','s','nexts','a','t'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 200
    
    % Collect data
%     [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    [ds, J] = collect_samples3(mdp, 5000, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    Q = mc_ret(data,mdp.gamma);
    
    % Get REPS weights
    [d, divKL] = solver.optimize(Q,data.phiV);
    
    % Print info
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    
    % Update pi
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);
    
    fprintf('%d) Entropy: %.4f,  Eta: %e,  KL (Weights): %.4f,  J: %e ', ...
        iter, entropy, solver.eta, divKL, J);
    if isa(policy,'Gaussian')
        fprintf(',  KL (Pi): %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;
    
end
