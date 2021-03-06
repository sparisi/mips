% Actor-critic policy gradient, as described by https://arxiv.org/pdf/1703.02660.pdf
% First, the generalized advantage A is estimated using V.
% Then, V is updated by minimizing the TD lambda error.
% Finally, the policy is updated by natural gradient on A.

% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
kl_bound = 0.05;
lambda_trace = 0.95;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;
omega = (rand(bfsV(),1)-0.5)*2;

data = [];
varnames = {'r','s','nexts','a','t'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 200
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    
    % Estimate A
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace);

    % Update V
    omega = fminunc(@(omega)mse_linear(omega,data.phiV,V+A), omega, options);

    % Re-estimate A with the updated critic
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace);

    % Estimate natural gradient
    dlogpi = policy.dlogPidtheta(data.s,data.a);
    grad = mean(bsxfun(@times,dlogpi,A),2);
    F = dlogpi * dlogpi' / length(A);
    rankF = rank(F);
    if rankF == size(F,1)
        grad_nat = F \ grad;
    else
        grad_nat = pinv(F) * grad;
    end
    stepsize = sqrt(kl_bound / (0.5*grad'*grad_nat));
    
    % Print info
    norm_g = norm(grad);
    norm_ng = norm(grad_nat);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   Norm (G): %e,   Norm (NG): %e,   J: %e', ...
        iter, entropy, norm_g, norm_ng, J);
    J_history(iter) = J;
    
    % Update pi
    policy_old = policy;
    policy = policy.update(policy.theta + grad_nat * stepsize);
    if isa(policy,'Gaussian')
        fprintf(',   KL (Pi): %.4f', kl_mvn2(policy, policy_old, policy.basis(data.s)));
    end
    fprintf('\n');
    
    iter = iter + 1;

end
