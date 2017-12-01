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
bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
% bfsV = @(varargin)basis_krbf(10, 20*[-ones(dim,1), ones(dim,1)], 0, varargin{:});
% bfsV = @(varargin)basis_rbf(5*[-ones(dim,1), ones(dim,1)], 0.5./[5; 5], 0, varargin{:});

solver = REPSavg_Solver(0.5,bfsV);

data = [];
varnames = {'r','s','nexts','a'};
bfsnames = { {'phiP', @(s)policy.basis_bias(s)}, {'phiV', bfsV} };
iter = 1;

%% Learning
while iter < 1500
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    nmax = size([ds.s],2);
    data = getdata(data,ds,nmax,varnames,bfsnames);

    % Ensure stationary distribution of transient dynamics by resetting 
    % the system with a probability 1/#steps.
    steps_ep = [ds.length];
    idx_init = [1 cumsum(steps_ep(1:end-1))+1];
    reset_prob = zeros(1,nmax);
    for i = 1 : episodes_learn
        reset_prob(idx_init(i):idx_init(i)+steps_ep(i)) = 1./steps_ep(i);
    end
    reset_prob(end) = [];

    [d, divKL] = solver.optimize(data.r, data.phiV, ...
        bsxfun(@plus, (1-reset_prob).*data.phiV_nexts, ... 
        reset_prob.*mean(data.phiV(:,idx_init),2)));

    policy = policy.weightedMLUpdate(d, data.a, data.phiP);

    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J_history(iter) = J(robj);
    fprintf('%d ) Entropy: %.2f, KL: %.2f, J: %.4f\n', iter, entropy, divKL, J(robj))
    
    iter = iter + 1;
    
    solver.plotV(mdp.stateLB, mdp.stateUB)
end

%%
plot(J_history)
show_simulation(mdp, policy.makeDeterministic, 1000, 0.01)