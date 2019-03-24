% COmpatible POlicy Search for discrete actions.
% https://arxiv.org/pdf/1902.02823.pdf

options_mse = optimoptions(@fminunc, ...
    'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-12, 'TolFun', 10^-12, 'MaxIter', 100);

options_dual = optimoptions('fmincon', ...
    'Algorithm', 'interior-point', ...
    'Display', 'off', ...
    'MaxFunEvals', 1000, ...
    'TolX', 10^-12, 'TolFun', 10^-12, 'MaxIter', 1000);

epsilon = 0.01;
beta = 0.01;
eta = 1;
omega = 1;
iter = 1;

%%
while iter < 250
    
    % Collect samples
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    s = [ds.s];
    a = [ds.a];
    
    % Estimate MC returns and fit R = psi*w to estimate the natural gradient w
    R = mc_ret(ds,mdp.gamma);
    R = R(robj,:);
    R = (R - mean(R)) / std(R); % standardize data
    psi = policy.dlogPidtheta(s,a);
    w = fminunc(@(w)mse_linear(w,psi,R), policy.theta, options_mse);

    % Compute A for all possible actions for all sampled states
    A = zeros(length(policy.action_list), size(s,2));
    for i = policy.action_list
        A(i,:) = w' * policy.dlogPidtheta(s, repmat(i, 1, size(s,2)));
    end

    % Compute current entropy for COPOS entropy bound
    probs_old = policy.distribution(s);
    probs_old(probs_old==0) = 1e-8;
    H_old = -mean(sum(probs_old.*log(probs_old),1));
    
    % Solve COPOS dual
    dual = @(x) x(1) * epsilon ...
        - x(2) * (H_old - beta) ...
        + (x(1) + x(2)) * mean(logsumexp((x(1) * log(probs_old) + A) / (x(1) + x(2)), 1));
    [x,~,~,info] = fmincon(dual, [eta;omega], ...
        [], [], [], [], [1e-8;1e-8], [1e8;1e8], [], options_dual);
    eta = x(1);
    omega = x(2);

    % Update and evaluate policy
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    policy_old = policy;
    policy = policy.update((eta * policy.theta + w) / (eta + omega));

    % Estimate KL
    probs_old = policy_old.distribution(s);
    probs = policy.distribution(s);
    KL = probs.*log(probs./probs_old);
    KL(isinf(KL) | isnan(KL)) = 0;
    KL = mean(sum(KL,1));    

    % Estimate entropy loss
    probs = policy.distribution(s);
    probs(probs==0) = 1e-8;
    H = -mean(sum(probs.*log(probs),1));
    H_diff = H_old - H;

    % Print info
    fprintf('%d) Entropy: %.2f,  Norm: %e,  J: %e,  Eta: %e,  Omega: %e,  KL: %e\n', ...
        iter, H_old, norm(w), J(robj), eta, omega, KL)
    J_history(iter) = J(robj);
    H_history(iter) = H_old;
    KL_history(iter) = KL;
    Hd_history(iter) = H_diff;
    
    iter = iter + 1;

end
