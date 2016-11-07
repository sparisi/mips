function [grad, stepsize] = GPOMDP(policy, data, gamma, lrate)
% Gradient of a Partially Observable Markov Decision Process.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Baxter and P L Bartlett
% Infinite-Horizon Policy-Gradient Estimation (2001)

episodeslength = [data.length];
totsteps = sum(episodeslength);
totepisodes = numel(data);
idx = zeros(1,totsteps);
idx(cumsum(episodeslength(1:end-1))+1) = 1;

sumdlog = cumsummove(policy.dlogPidtheta([data.s],[data.a]),idx);
grad = sumdlog * [data.gammar]';

if gamma == 1
    grad = grad / totsteps;
else
    grad = grad / totepisodes;
end

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end
