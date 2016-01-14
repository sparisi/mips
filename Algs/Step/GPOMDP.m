function [grad, stepsize] = GPOMDP(policy, data, gamma, lrate)
% Gradient of a Partially Observable Markov Decision Process.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Baxter and P L Bartlett
% Infinite-Horizon Policy-Gradient Estimation (2001)

dlogpi = policy.dlogPidtheta(horzcat(data.s),horzcat(data.a));
episodeslength = horzcat(data.length);
totstep = sum(episodeslength);
totepisodes = numel(data);
dreward = size(data(1).r,1);
reward = horzcat(data.r);

grad = zeros(policy.dparams,dreward);
for i = 1 : totepisodes
    idx1 = sum(episodeslength(1:i-1))+1;
    idx2 = idx1 + episodeslength(i)-1;
    sumdlogpi = cumsum(dlogpi(:,idx1:idx2),2);
	discountedrew = bsxfun(@times,reward(:,idx1:idx2),gamma.^([1:data(i).length]-1));
	grad = grad + sumdlogpi * discountedrew';
end
    
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