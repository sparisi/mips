function [grad, stepsize] = REINFORCE_C(policy, data, gamma, lrate)
% As REINFORCE, but instead of using Monte-Carlo estimates of the return, 
% a critic is learned with compatible features.
% Q(s,a) = phi(s,a)*w
% where phi(s,a) is the derivative of log(pi) wrt the policy parameters.
%
% =========================================================================
% REFERENCE
% R Sutton
% Policy Gradient Methods for Reinforcement Learning with Function
% Approximation (1999)

R = mc_ret(data,gamma);
R = (R - mean(R)) / std(R); % standardize data
dlogpi = policy.dlogPidtheta([data.s],[data.a]);

options = optimoptions(@fminunc, 'Algorithm', 'trust-region', ...
    'GradObj', 'on', ...
    'Display', 'off', ...
    'MaxFunEvals', 100, ...
    'Hessian', 'on', ...
    'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);
w = fminunc(@(w)mse_linear(w,dlogpi,R), zeros(size(dlogpi,1),1), options);

R = w'*dlogpi;
grad = dlogpi * R' / size(R,2);
grad = mean(grad,2);

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end