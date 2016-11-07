function h = HessianRF(policy, data, gamma)
% Computes the Hessian of a policy wrt its parameters.
% H is a [D x D x R] matrix, where D is the length of the policy parameters
% and R is the number of immediate rewards received at each time step. Each 
% page of H corresponds to the Hessian wrt an objective.
%
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, M Restelli
% Multi-objective Reinforcement Learning through Continuous Pareto Manifold 
% Approximation (2016)

actions = horzcat(data.a);
states = horzcat(data.s);
dlogpi = policy.dlogPidtheta(states,actions);
hlogpi = policy.hlogPidtheta(states,actions);
episodeslength = horzcat(data.length);
totstep = sum(episodeslength);
totepisodes = numel(data);

sumdlog = cumsumidx(dlogpi,cumsum(episodeslength));
sumdlog2 = bsxfun(@times,permute(sumdlog,[1 3 2]),permute(sumdlog,[3 1 2]));
sumhlog = cumsumidx3(hlogpi,cumsum(episodeslength));
sumrew = cumsumidx(horzcat(data.gammar),cumsum(episodeslength));

h = squeeze( sum( bsxfun(@times, sumdlog2 + sumhlog, reshape(sumrew',[1 1 size(sumrew')])), 3) );

if gamma == 1
    h = h / totstep;
else
    h = h / totepisodes;
end

end