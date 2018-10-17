% Proximal policy optimization https://arxiv.org/abs/1707.06347

% To learn V
options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

mdp.gamma = 0.99;
lrate = 0.05;
lambda_trace = 0.95;
e_clip = 0.2;
batch_size = 64;
max_epochs = 20;

bfsV = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});
% bfsV = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});
bfsV = bfs;

omega = (rand(bfsV(),1)-0.5)*2;

data = [];
varnames = {'r','s','nexts','a','endsim'};
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
    omega = fminunc(@(omega)learn_V(omega,data.phiV,A+V), omega, options);

    % Estimate gradient
    old_probs = policy.evaluate(data.a, data.s);
    for epoch = 1 : max_epochs
        ratio = policy.evaluate(data.a, data.s) ./ old_probs;
        clipped = min(max(ratio, 1-e_clip), 1+e_clip);
        idx = ratio.*A <= clipped.*A;
        ratio(~idx) = 0; % Gradient of clip(ratio)*A is 0
        dlogpi = policy.dlogPidtheta(data.s,data.a);
        grad = mean(bsxfun(@times,dlogpi,ratio.*A),2);
        norm_g = norm(grad);
        policy = policy.update(policy.theta + lrate*grad/norm(grad));
    end

    % Print info
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    fprintf('%d) Entropy: %.2f,   Norm: %e,   J: %e \n', ...
        iter, entropy, norm_g, J);
    J_history(iter) = J;
    
    iter = iter + 1;
end


%%
function [g, gd, h] = learn_V(omega, Phi, T)
% Mean squared TD error
V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
h = Phi*Phi'/size(T,2);
end
