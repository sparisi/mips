% Deterministic Policy Gradient on bandits.
%
% =========================================================================
% REFERENCE
% D Silver, G Lever, N Heess, T Degris, D Wierstra, M Riedmiller
% Deterministic Policy Gradient Algorithms (2014)

clear, clear global
close all

dim = 10;
N = 100;
rfun = @(x) quadcost(x, 4*ones(dim,1)); sigma = 16;
% rfun = @(x) rosenbrock(x); sigma = 0.1;
% rfun = @(x) rastrigin(x); sigma = 0.1;
% rfun = @(x) noisysphere(x); sigma = 0.1;

lrate_theta = 0.001;
lrate_v = 0.0001;
lrate_w = 0.0001;
iter = 1;


%% Setup actor and critic
% Policy
theta = rand(dim,1);
d_policy = eye(dim); % Derivative of the policy wrt theta

% Q-function (with compatible function approximation)
w = zeros(size(theta));
Qfun = @(a,w,theta) (bsxfun(@minus,a,theta)' * d_policy' * w)';
features = @(a,theta) d_policy * bsxfun(@minus,a,theta);


%% Learn
while iter < 100000
    
%     sigma = sigma * 0.999;
    
%     updateplot('Theta',iter,theta)

    noise = mvnrnd(zeros(dim,1),sigma*eye(dim),N)';
    action = bsxfun(@plus,theta,noise);
    reward = rfun(action);
    
    % Actor and critic update
    delta = reward - Qfun(action, w, theta);
    theta = theta + lrate_theta * w;
    w     = w     + lrate_w     * features(action,theta) * delta' / N;
    
    if any(isnan(theta)) || any(isnan(w)) || ...
            any(isinf(theta)) || any(isinf(w)) 
        error('Inf or NaN.')
    end
    
    R_eval = rfun(theta);
    Q_eval = Qfun(theta, w, theta);
    updateplot('Error', iter, (R_eval-Q_eval)^2, 1)
    updateplot('Reward', iter, R_eval, 1)
    if iter == 1, autolayout, end
    
    iter = iter + 1;
    
end
