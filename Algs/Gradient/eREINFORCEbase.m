function [grad, stepsize] = eREINFORCEbase(policy, data, gamma, lrate)
% REINFORCE with optimal baseline.
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

% Compute optimal baseline
bden = sum(sumdlog.^2,2);
bnum = sumdlog.^2 * sumrew';
baseline = bsxfun(@times,bnum,1./bden);
baseline(isnan(baseline)) = 0; % When 0 / 0

% Compute gradient
bterm = bsxfun(@times,permute(sumdlog,[3 1 2]),baseline');
bterm = sum(bterm,3);
grad = sumdlog * sumrew' - bterm';

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
