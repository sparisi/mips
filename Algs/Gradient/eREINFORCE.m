function [grad, stepsize] = eREINFORCE(policy, data, gamma, lrate)
% REward Increment = Nonnegative Factor times Offset Reinforcement times 
% Characteristic Eligibility.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% R J Williams
% Simple Statistical Gradient-Following Algorithms for Connectionist 
% Reinforcement Learning (1992)

episodeslength = [data.length];
totsteps = sum(episodeslength);
totepisodes = numel(data);
idx = cumsum(episodeslength);

sumdlog = cumsumidx(policy.dlogPidtheta([data.s],[data.a]),idx);
gammar = bsxfun(@times, [data.r], gamma.^([data.t]-1));
sumrew = cumsumidx(gammar,idx);

grad = sumdlog * sumrew';

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