% Trust Region Policy Optimization https://arxiv.org/pdf/1502.05477.pdf
% Like ACPG, but 
%   1) Instead of GRAD_NAT = F \ GRAD, the natural gradient is computed by 
%      the conjugate gradient method, knowing that F * GRAD_NAT = GRAD; and
%   2) The actual stepsize is computed by line seach, to check that the KL
%      bound is really satisfied.

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
while iter < 2000
    
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
    [grad_nat,~,~,~,~] = pcg(F,grad,1e-10,50); % Use conjugate gradient (~ are to avoid messages)
%     [grad_nat,~,~,~,~] = cgs(F,grad,1e-10,50);

    % Line search
    stepsize = sqrt(kl_bound / (0.5*grad'*grad_nat));
    obj_fun = @(x) mean(policy.update(x).logpdf([data.a],[data.s]).*A);
    kl = @(x) kl_mvn2(policy.update(x), policy, policy.basis(data.s));
    [success, theta, n_back] = linesearch(obj_fun, policy.theta, stepsize*grad_nat, grad'*grad_nat*stepsize, kl, kl_bound);
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
