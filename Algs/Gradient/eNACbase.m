function [grad_nat, stepsize] = eNACbase(policy, data, gamma, epsilon)
% Episodic Natural Actor Critic with optimal baseline.
% GRAD_NAT is a [D x R] matrix, where D is the length of the gradient and R
% is the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Peters, S Vijayakumar, S Schaal
% Natural Actor-Critic (2008)

totepisodes = numel(data);

idx = cumsum([data.length]);
ii = zeros(1,sum([data.length]));
ii(idx(1:end-1)+1) = 1;
timesteps = cumsummove(ones(1,sum([data.length])),ii);

allgamma = gamma.^(timesteps-1);

sumdlog = cumsumidx(bsxfun(@times,policy.dlogPidtheta([data.s],[data.a]),allgamma),idx);
sumdlog = [sumdlog; ones(1,totepisodes)];
F = sumdlog * sumdlog';
sumrew = cumsumidx(bsxfun(@times,[data.r],allgamma),idx);
g = sumdlog * sumrew';
aR = sum(sumrew,2);
el = sum(sumdlog,2);

F = F / totepisodes;
g = g / totepisodes;
el = el / totepisodes;
aR = aR / totepisodes;

rankF = rank(F);
if rankF == size(F,1) + 1
    Q = 1 / totepisodes * (1 + el' / (totepisodes * F - el * el') * el);
    b = Q * (aR' - el' / F * g);
    grad_nat = F \ (g - el * b);
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, size(F,1) + 1);
    Q = 1 / totepisodes * (1 + el' * pinv(totepisodes * F - el * el') * el);
    b = Q * (aR' - el' * pinv(F) * g);
    grad_nat = pinv(F) * (g - el * b);
end

if nargin == 4
    lambda = sqrt(diag(g' * grad_nat) / (8 * epsilon));
    lambda = max(lambda,1e-8);
    stepsize = 1 ./ (2 * lambda)';
end

grad_nat = grad_nat(1:end-1,:);
