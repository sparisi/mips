function [grad, stepsize] = REINFORCE(policy, data, gamma, lrate)
% Basic version of REINFORCE with mean(R) as baseline.
%
% =========================================================================
% REFERENCE
% R J Williams
% Simple Statistical Gradient-Following Algorithms for Connectionist 
% Reinforcement Learning (1992)

R = mc_ret(data,gamma);
R = R - mean(R,2);
dlogpi = policy.dlogPidtheta([data.s],[data.a]);
grad = dlogpi * R' /size(R,2);
grad = mean(grad,2);

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end