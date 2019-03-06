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
w = linear_regression(dlogpi, R);

R = w'*dlogpi;
grad = dlogpi * R' / size(R,2);
grad = mean(grad,2);

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end