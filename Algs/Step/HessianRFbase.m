function h = HessianRFbase(policy, data, gamma)
% Computes the Hessian of a policy wrt its parameters using the optimal
% baseline.
% H is a [D x D x R] matrix, where D is the length of the policy parameters
% and R is the number of immediate rewards received at each time step. Each 
% page of H corresponds to the Hessian wrt an objective.
%
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, M Restelli
% Multi-objective Reinforcement Learning through Continuous Pareto Manifold 
% Approximation (2016)

actions = [data.a];
states = [data.s];
dlogpi = policy.dlogPidtheta(states,actions);
hlogpi = policy.hlogPidtheta(states,actions);
episodeslength = [data.length];
totstep = sum(episodeslength);
totepisodes = numel(data);
dreward = size(data(1).r,1);

sumdlog = cumsumidx(dlogpi,cumsum(episodeslength));
sumdlog2 = bsxfun(@times,permute(sumdlog,[1 3 2]),permute(sumdlog,[3 1 2]));
sumhlog = cumsumidx3(hlogpi,cumsum(episodeslength));
gammar = bsxfun(@times, [data.r], gamma.^([data.t]-1));
sumrew = cumsumidx(gammar,cumsum(episodeslength));

% Compute the optimal baseline
tmp = (sumdlog2 + sumhlog).^2;
bden = sum(tmp,3);
bden = repmat(bden,1,1,dreward);
bnum = squeeze( sum( bsxfun(@times, tmp, reshape(sumrew',[1 1 size(sumrew')])), 3) );
b = bnum ./ bden;
b(isnan(b)) = 0; % When 0 / 0

% Compute the Hessian
rewb = bsxfun(@plus, -b, reshape(sumrew,[1 1 size(sumrew)]));
sumhd = sumdlog2 + sumhlog;
h = squeeze( sum( repmat(sumhd,1,1,1,dreward).*permute(rewb,[1 2 4 3]), 3) );

if gamma == 1
    h = h / totstep;
else
    h = h / totepisodes;
end

end