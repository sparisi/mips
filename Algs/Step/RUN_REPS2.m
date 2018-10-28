% This version of REPS never ends an episode unless a terminal state is
% reached or a reset occurs. See COLLECT_SAMPLES_INF for more info.

reset_prob = 0.01;

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
    
    [ds, J] = collect_samples_inf(mdp, steps_learn*episodes_learn, reset_prob, policy);
    entropy = policy.entropy([ds.s]);

    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    idx_init = [1 find(data.endsim)+1];
    idx_init(end) = [];

    [d, divKL] = solver.optimize(data.r, data.phiV, ...
        bsxfun(@plus, (1-reset_prob).*data.phiV_nexts, ... 
        reset_prob.*mean(data.phiV(:,idx_init),2)));
    
    policy_old = policy;
    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    fprintf('%d ) Entropy: %.3f,  Eta: %e,  KL (Weights): %.4f,  J: %e', ...
        iter, entropy, solver.eta, divKL, J)
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