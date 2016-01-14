function [grad_nat, stepsize] = eNACbase(policy, data, gamma, lrate)
% Episodic Natural Actor Critic with optimal baseline.
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
aR = sum(rewgamma,2);
el = sum(sumdlog,2);

F = F / totepisodes;
g = g / totepisodes;
el = el / totepisodes;
aR = aR / totepisodes;

rankF = rank(F);
if rankF == dparams + 1
    Q = 1 / totepisodes * (1 + el' / (totepisodes * F - el * el') * el);
    b = Q * (aR' - el' / F * g);
    grad_nat = F \ (g - el * b);
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, dparams+1);
    Q = 1 / totepisodes * (1 + el' * pinv(totepisodes * F - el * el') * el);
    b = Q * (aR' - el' * pinv(F) * g);
    grad_nat = pinv(F) * (g - el * b);
end

if nargin == 4
    lambda = sqrt(diag(g' * grad_nat) / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 ./ (2 * lambda)';
end

grad_nat = grad_nat(1:end-1,:);
