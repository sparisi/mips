function [grad_nat, stepsize] = NaturalPG(alg, policy, data, gamma, lrate)
% GRAD_NAT is a [D x R] matrix, where D is the length of the gradient and R
% is the number of immediate rewards received at each time step.

dlogpi = policy.dlogPidtheta(horzcat(data.s),horzcat(data.a));
episodeslength = horzcat(data.length);
totepisodes = numel(data);
dparams = policy.dparams;

timesteps = cell2mat(arrayfun(@(x)1:x,episodeslength,'uni',0));
allgamma = gamma.^(timesteps-1);
sumdlog = cumsumidx(bsxfun(@times,dlogpi,allgamma),cumsum(episodeslength));
F = sumdlog * sumdlog';

F = F / totepisodes;

switch alg
    case 'r'
        grad = eREINFORCE(policy, data, gamma);
    case 'rb'
        grad = eREINFORCEbase(policy, data, gamma);
    case 'g'
        grad = GPOMDP(policy, data, gamma);
    case 'gb'
        grad = GPOMDPbase(policy, data, gamma);
    otherwise
        error('Unknown algoritm.');
end

if rank(F) == policy.dparams
    grad_nat = F \ grad;
else
	warning('Fisher matrix is lower rank (%d instead of %d).', rank(F), dparams);
    grad_nat = pinv(F) * grad;
end

if nargin == 5
    lambda = sqrt(diag(grad' * grad_nat) / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 ./ (2 * lambda');
end