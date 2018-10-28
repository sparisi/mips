% https://arxiv.org/pdf/1502.05477.pdf
% It is basically ACPG, but the stepsize is computed by line seach.
% (For linear approximator with few parameters, we do not have to use
% conjugate gradient to approximate the natural gradient as in the paper, 
% since inverting the FIM is not excessively hard).

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
varnames = {'r','s','nexts','a','endsim','terminal'};
bfsnames = { {'phiP', @(s)policy.get_basis(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 200
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).endsim(end) = 1; % To separate episodes for GAE
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    
    % Estimate A
    V = omega'*data.phiV;
    A = gae(data,V,mdp.gamma,lambda_trace);

    % Update V
    omega = fminunc(@(omega)mstde_v(omega,data.phiV,V+A), omega, options);
    
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

    % Line search
    stepsize = sqrt(kl_bound / (0.5*grad'*grad_nat));
    max_adv = @(theta) mean(policy.update(theta).logpdf([data.a],[data.s]).*A);
    kl = @(theta) kl_mvn2(policy.update(theta), policy, policy.basis(data.s));
    [success, theta, n_back] = linesearch(max_adv, policy.theta, stepsize*grad_nat, grad'*grad_nat*stepsize, kl, kl_bound);
    if ~success, warning('Could not satisfy the KL constraint.'), end % in this case, theta = policy.theta
    
    % Print info
    norm_g = norm(grad);
    norm_ng = norm(grad_nat);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   Norm (G): %e,   Norm (NG): %e,   J: %e \n', ...
        iter, entropy, norm_g, norm_ng, J);
    J_history(iter) = J;
    
    % Update pi
    policy = policy.update(theta);
    
    iter = iter + 1;

end
