% Step-based REPS for average reward MDPs.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)
%
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss (2017)

rng(1)

mdp_avg = MDP_avg(mdp,0.02);

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
bfsV = @(varargin)basis_krbf(6, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

solver = REPS_Solver2(0.1,bfsV);
solver.verbose = 0;

data = [];
varnames = {'r','s','nexts','a','endsim'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1000
    
    [ds, J] = collect_samples(mdp_avg, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).endsim(end) = 1;
    end
    entropy = policy.entropy([ds.s]);

    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    idx_init = [1 find(data.endsim)+1];
    idx_init(end) = [];

    [d, divKL] = solver.optimize(data.r, data.phiV, ...
        bsxfun(@plus, (1-mdp_avg.reset_prob).*data.phiV_nexts, ... 
        mdp_avg.reset_prob.*mean(data.phiV(:,idx_init),2)));
    
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    fprintf('%d ) Entropy: %.3f,  KL (Weights): %.4f,  J: %e', iter, entropy, divKL, J)
    if isa(policy,'Gaussian')
        fprintf(',  KL: %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;
    
    %%
    if solver.verbose
        solver.plotV(mdp.stateLB, mdp.stateUB)
        policy.plotmean(mdp.stateLB, mdp.stateUB)
    end
    
end

%%
figure
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 1000, .01)