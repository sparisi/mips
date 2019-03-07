function [grad, stepsize] = GPOMDP(policy, data, gamma, lrate)
% Gradient of a Partially Observable Markov Decision Process.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Baxter and P L Bartlett
% Infinite-Horizon Policy-Gradient Estimation (2001)

totsteps = size([ds.r],2);
totepisodes = numel(data);

sumdlog = cumsummove(policy.dlogPidtheta([data.s],[data.a]),[data.t]==1);
if gamma < 1
    grad = sumdlog * bsxfun(@times, [data.r], gamma.^([data.t]-1))';
end 

if gamma == 1
    grad = grad / totsteps; % Avg reward MDP
else
    grad = grad / totepisodes;
end

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end
