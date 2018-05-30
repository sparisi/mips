% Step-based REPS.
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
bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

% bfsV = @(varargin)basis_krbf(10, 20*[-ones(dim,1), ones(dim,1)], 0, varargin{:});
% bfsV = @(varargin)basis_rbf(5*[-ones(dim,1), ones(dim,1)], 0.5./[5; 5], 0, varargin{:});

solver = REPSavg_Solver(0.1,bfsV);

data = [];
varnames = {'r','s','nexts','a'};
bfsnames = { {'phiP', @(s)policy.basis_bias(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 5; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

mean_V_init = 0; % Keep a running mean of the initial state V
n = 0;

%% Learning
while iter < 100
    
    [ds, J] = collect_samples(mdp_avg, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);

    idx_init = cumsum([ds.length])+1;
    idx_init(end) = [];
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);

    n = n + length(idx_init); % Update running mean count
    mean_V_init = mean_V_init + sum(data.phiV(:,idx_init),2); % Update running mean

    [d, divKL] = solver.optimize(data.r, data.phiV, ...
        bsxfun(@plus, (1-mdp_avg.reset_prob).*data.phiV_nexts, ... 
        mdp_avg.reset_prob.*mean_V_init/n));

    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J;
    fprintf('%d ) Entropy: %.2f,  KL: %.2f,  J: %e\n', iter, entropy, divKL, J)
    
    iter = iter + 1;
    
    solver.plotV(mdp.stateLB, mdp.stateUB)
end

%%
figure
plot(J_history)