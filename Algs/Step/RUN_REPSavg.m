% Step-based REPS.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)
%
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss (2017)

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
bfsV = @(varargin)basis_krbf(10, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
% bfsV = @(varargin)basis_krbf(10, 20*[-ones(dim,1), ones(dim,1)], 0, varargin{:});
% bfsV = @(varargin)basis_rbf(5*[-ones(dim,1), ones(dim,1)], 0.5./[5; 5], 0, varargin{:});

solver = REPSavg_Solver(0.5,bfsV);

nmax = episodes_learn * steps_learn;
data = [];
varnames = {'r','s','nexts','a'};
bfsnames = { {'phiP', @(s)policy.basis1(s)}, {'phiV', bfsV} };
iter = 1;

%% Learning
while iter < 1500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    data = getdata(data,ds,nmax,varnames,bfsnames);

    % Ensure stationary distribution of transient dynamics by resetting 
    % the system with a probability
    reset_prob = 1/steps_learn; 

    [d, divKL] = solver.optimize(data.r, data.phiV, ...
        bsxfun(@plus, (1-reset_prob)*data.phiV_nexts, ... 
        reset_prob*mean(data.phiV(:,1:steps_learn:end),2)));

    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f, KL: %.2f, J: %.4f\n', iter, entropy, divKL, J(robj))
    
    iter = iter + 1;
end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 100, 0.01)