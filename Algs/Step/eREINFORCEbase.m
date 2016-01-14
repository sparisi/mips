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

dlogpi = policy.dlogPidtheta(horzcat(data.s),horzcat(data.a));
episodeslength = horzcat(data.length);
totstep = sum(episodeslength);
totepisodes = numel(data);

sumdlog = cumsumidx(dlogpi,cumsum(episodeslength));
sumrew = cumsumidx(horzcat(data.gammar),cumsum(episodeslength));

% Compute optimal baseline
bden = sum(sumdlog.^2,2);
bnum = sum(bsxfun(@times,sumdlog.^2,reshape(sumrew',[1 size(sumrew')])),2);
bnum = squeeze(bnum);
b = bsxfun(@times,bnum,1./bden);
b(isnan(b)) = 0; % When 0 / 0

% Compute gradient
grad = sum( bsxfun( @times, sumdlog, ...
    permute( bsxfun( @plus,reshape(sumrew,[1 size(sumrew)]),-b ), [1 3 2]) ...
    ), 2);
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
