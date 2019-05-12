% Step-based REPS for average reward MDPs.
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)
%
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss (2017)

reset_prob = 0.01;
mdp_avg = MDP_avg(mdp,reset_prob);

% bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(10, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

epsilon = 0.01;
solver = REPS_Solver(epsilon,bfsV);
solver = REPS_Solver2(epsilon,bfsV);
solver.verbose = 1;

data = [];
varnames = {'r','s','nexts','a','t'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1000
    
    % If you use mdp_avg an episode ends after steps_learn of for a random reset
    % If you use collect_samples_inf there is no time limit (an episode ends only because of a random reset)

%     [ds, J] = collect_samples3(mdp_avg, 5000, steps_learn, policy);
    [ds, J] = collect_samples_inf(mdp, 5000, reset_prob, policy);
    entropy = policy.entropy([ds.s]);

    avg_reset = length([ds.length]) / sum([ds.length]);
    if avg_reset < reset_prob*0.9
        warning(['Reset probability too low (' num2str(avg_reset) ').'])
    elseif avg_reset > 1.1*reset_prob
        warning(['Reset probability too high (' num2str(avg_reset) ').'])
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
        fprintf(',  KL (Policy): %.4f', kl_mvn2(policy_old, policy, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;
    
    %%
    if solver.verbose && mdp.dstate == 2
        updatescatter('Weights (s)', data.s(1,:), data.s(2,:), [], d, 50)
        if mdp.daction == 1
            updatescatter('Weights (s,a)', data.s(1,:), data.s(2,:), data.a, d, 50)
            A = data.r + solver.theta' * (phiVN - data.phiV);
            updatescatter('Advantage', data.s(1,:), data.s(2,:), data.a, A, 50)
        end
        solver.plotV(mdp.stateLB, mdp.stateUB, 'surf')
        policy.plotmean(mdp.stateLB, mdp.stateUB, mdp.actionLB, mdp.actionUB)
        if iter == 2, autolayout, end
    end
        
end

%%
show_simulation(mdp, policy.makeDeterministic, 200, 0.01)