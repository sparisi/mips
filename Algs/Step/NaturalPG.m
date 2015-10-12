function [dJdtheta, stepsize] = NaturalPG(alg, policy, data, gamma, robj, lrate)

dlogpi_r = policy.dlogPidtheta;
fisher = zeros(dlogpi_r,dlogpi_r);

num_trials = max(size(data));
parfor trial = 1 : num_trials
	for step = 1 : max(size(data(trial).a)) 
		loggrad = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        fisher = fisher + loggrad * loggrad';
	end
end
fisher = fisher / num_trials;

if strcmp(alg, 'r')
    grad = eREINFORCE(policy, data, gamma, robj);
elseif strcmp(alg, 'rb')
    grad = eREINFORCEbase(policy, data, gamma, robj);
elseif strcmp(alg, 'g')
    grad = GPOMDP(policy, data, gamma, robj);
elseif strcmp(alg, 'gb')
    grad = GPOMDPbase(policy, data, gamma, robj);
else
    error('Unknown algoritm.');
end

if rank(fisher) == dlogpi_r
    dJdtheta = fisher \ grad;
else
	warning('Fisher matrix is lower rank (%d instead of %d).', rank(fisher), dlogpi_r);
    dJdtheta = pinv(fisher) * grad;
end

if nargin >= 6
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end