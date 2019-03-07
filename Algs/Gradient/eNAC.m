function [grad_nat, stepsize] = eNAC(policy, data, gamma, lrate)
% Episodic Natural Actor Critic.
% GRAD_NAT is a [D x R] matrix, where D is the length of the gradient and R
% is the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Peters, S Vijayakumar, S Schaal
% Natural Actor-Critic (2008)

dlogpi = policy.dlogPidtheta([data.s],[data.a]);
totepisodes = numel(data);

idx = cumsum([data.length]);

allgamma = gamma.^([data.t]-1);

sumdlog = cumsumidx(bsxfun(@times,dlogpi,allgamma),idx);
sumdlog = [sumdlog; ones(1,totepisodes)];
F = sumdlog * sumdlog';
gammar = bsxfun(@times, [data.r], gamma.^([data.t]-1));
sumrew = cumsumidx(gammar,idx);
g = sumdlog * sumrew';

F = F / totepisodes;
g = g / totepisodes;

rankF = rank(F);
if rankF == size(F,1) + 1
	grad_nat = F \ g;
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, size(F,1) + 1);
	grad_nat = pinv(F) * g;
end

if nargin == 4
    lambda = sqrt(diag(g' * grad_nat) / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 ./ (2 * lambda)';
end

grad_nat = grad_nat(1:end-1,:);
