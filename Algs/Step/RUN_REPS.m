% Step-based REPS for average reward MDPs.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)
%
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss (2017)

reset_prob = 0.1;
mdp_avg = MDP_avg(mdp,reset_prob);

% bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(6, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

solver = REPS_Solver(0.1,bfsV);
solver.verbose = 1;

data = [];
varnames = {'r','s','nexts','a','t'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1000
    
%     [ds, J] = collect_samples3(mdp_avg, 15000, 500, policy);
    % [ds, J] = collect_samples(mdp_avg, episodes_learn, steps_learn, policy);
    [ds, J] = collect_samples_inf(mdp, 10000, reset_prob, policy);
    entropy = policy.entropy([ds.s]);

    avg_reset = length([ds.length]) / sum([ds.length]);
    if avg_reset < reset_prob*0.9 || avg_reset > 1.1
        warning('Reset probability not as expected.')
    end

    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    
    phiVN = bsxfun(@plus, (1-reset_prob).*data.phiV_nexts, reset_prob.*mean(data.phiV(:,data.t==1),2));
    [d, divKL] = solver.optimize(data.r, data.phiV, phiVN);
    
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    fprintf('%d ) Entropy: %.3f,  Eta: %e,  KL (Weights): %.4f,  J: %e', ...
        iter, entropy, solver.eta, divKL, J)
    if isa(policy,'Gaussian')
        fprintf(',  KL (Policy): %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;
    
    %%
    if solver.verbose && mdp.dstate == 2
        solver.plotV(mdp.stateLB, mdp.stateUB)
        policy.plotmean(mdp.stateLB, mdp.stateUB)
        autolayout
    end
    
end

%%
show_simulation(mdp, policy.makeDeterministic, 200, 0.01)
