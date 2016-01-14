function [grad_nat, stepsize] = eNAC(policy, data, gamma, lrate)
% Episodic Natural Actor Critic.
% GRAD_NAT is a [D x R] matrix, where D is the length of the gradient and R
% is the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Peters, S Vijayakumar, S Schaal
% Natural Actor-Critic (2008)

dlogpi = policy.dlogPidtheta(horzcat(data.s),horzcat(data.a));
episodeslength = horzcat(data.length);
totepisodes = numel(data);
dparams = policy.dparams;

timesteps = cell2mat(arrayfun(@(x)1:x,episodeslength,'uni',0));
allgamma = gamma.^(timesteps-1);
sumdlog = cumsumidx(bsxfun(@times,dlogpi,allgamma),cumsum(episodeslength));
sumdlog = [sumdlog; ones(1,totepisodes)];
F = sumdlog * sumdlog';
rewgamma = cumsumidx(bsxfun(@times,horzcat(data.r),allgamma),cumsum(episodeslength));
g = sumdlog * rewgamma';

F = F / totepisodes;
g = g / totepisodes;

rankF = rank(F);
if rankF == dparams + 1
	grad_nat = F \ g;
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, dparams+1);
	grad_nat = pinv(F) * g;
end

if nargin == 4
    lambda = sqrt(diag(g' * grad_nat) / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 ./ (2 * lambda)';
end

grad_nat = grad_nat(1:end-1,:);
