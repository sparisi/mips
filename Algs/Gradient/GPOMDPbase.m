function [grad, stepsize] = GPOMDPbase(policy, data, gamma, lrate)
% Gradient of a Partially Observable Markov Decision Process with optimal
% baseline.
% GRAD is a [D x R] matrix, where D is the length of the gradient and R is
% the number of immediate rewards received at each time step.
%
% =========================================================================
% REFERENCE
% J Baxter and P L Bartlett
% Infinite-Horizon Policy-Gradient Estimation (2001)

dlogpi = policy.dlogPidtheta([data.s],[data.a]);
episodeslength = [data.length];
totstep = sum(episodeslength);
totepisodes = numel(data);
dreward = size(data(1).r,1);

% Compute optimal baseline
bnum = zeros(policy.dparams, max(episodeslength), dreward);
bden = zeros(policy.dparams, max(episodeslength));
for i = 1 : totepisodes
    idx1 = sum(episodeslength(1:i-1))+1;
    idx2 = idx1 + episodeslength(i)-1;
    sumdlog2 = cumsum(dlogpi(:,idx1:idx2),2).^2;
    steps = data(i).length;
    data(i).gammar = bsxfun(@times, data(i).r, gamma.^(data(i).t-1));
    bden(:,1:steps) = bden(:,1:steps) + sumdlog2;
    bnum(:,1:steps,:) = bnum(:,1:steps,:) + ...
        bsxfun(@times,sumdlog2,reshape(data(i).gammar',[1 size(data(i).gammar')]));
end
bden = repmat(bden,1,1,dreward);
b = bnum ./ bden;
b(isnan(b)) = 0; % When 0 / 0

% Compute gradient
grad = zeros(policy.dparams,dreward);
for i = 1 : totepisodes
    idx1 = sum(episodeslength(1:i-1))+1;
    idx2 = idx1 + episodeslength(i)-1;
    sumdlog = cumsum(dlogpi(:,idx1:idx2),2);
    steps = data(i).length;
    grad = grad + squeeze(sum(bsxfun(@times, sumdlog, bsxfun(@plus,permute(data(i).gammar,[3 2 1]),-b(:,1:steps,:))),2));
end
    
if gamma == 1
    grad = grad / totstep; % Avg reward MDP
else
    grad = grad / totepisodes;
end

if nargin == 4
    normgrad = matrixnorms(grad,2);
    lambda = max(normgrad,1e-8); % to avoid numerical problems
    stepsize = sqrt(lrate) ./ lambda;
end