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

dlogpi = policy.dlogPidtheta(horzcat(data.s),horzcat(data.a));
episodeslength = horzcat(data.length);
totstep = sum(episodeslength);
totepisodes = numel(data);

sumdlog = cumsumidx(dlogpi,cumsum(episodeslength));
sumrew = cumsumidx(horzcat(data.gammar),cumsum(episodeslength));

grad = sum(bsxfun(@times,sumdlog,reshape(sumrew',[1 size(sumrew')])),2);
grad = squeeze(grad);

if gamma == 1
    grad = grad / totstep;
else
    grad = grad / totepisodes;
end

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end