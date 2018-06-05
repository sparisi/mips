% Actor-critic policy gradient, as described by https://arxiv.org/pdf/1703.02660.pdf
% First, the generalized advantage A is estimated using V.
% Then, V is updated by minimizing the TD-error (targets are Monte-Carlo estimates).
% Finally, the policy is updated by natural gradient on A.

rng(1)

% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

lrate = 0.05;
gamma = mdp.gamma;
lambda_trace = 0.95;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

omega = (rand(bfsV(),1)-0.5)*2;

data = [];
varnames = {'r','s','nexts','a','endsim','Q'};
bfsnames = { {'phiP', @(s)policy.basis_bias(s)}, {'phiV', bfsV} };
iter = 1;

max_reuse = 1; % Reuse all samples from the past X iterations
max_samples = zeros(1,max_reuse);

%% Learning
while iter < 1000
    
    % Collect data
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    for i = 1 : numel(ds)
        ds(i).endsim(end) = 1;
    end
    entropy = policy.entropy([ds.s]);
    max_samples(mod(iter-1,max_reuse)+1) = size([ds.s],2);
    data = getdata(data,ds,sum(max_samples),varnames,bfsnames);
    
    % Estimate A
    V = omega'*data.phiV;
    A = gae(data,V,gamma,lambda_trace);

    % Update V
    omega = fminunc(@(omega)learn_V(omega,data.phiV,data.Q), omega, options);
    
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
    stepsize = sqrt(lrate / (grad'*grad_nat));
    
    % Print info
    norm_g = norm(grad);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   Norm: %e,   J: %e \n', ...
        iter, entropy, norm_g, J);
    J_history(iter) = J;
    
    % Update pi
    policy = policy.update(policy.theta + grad_nat * stepsize);
    
    iter = iter + 1;
end


%%
function [g, gd] = learn_V(omega, Phi, T)
% Mean squared TD error
V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
end
