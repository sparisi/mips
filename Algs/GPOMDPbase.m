%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, stepsize] = GPOMDPbase(policy, data, gamma, robj, lrate)

dlp = policy.dlogPidtheta;
dJdtheta = zeros(dlp,1);

totstep = 0;

% Compute baselines
num_trials = max(size(data));

actions = cell(1,numel(data));
[actions{:}] = data.a;
lengths = cellfun('length',actions);

bnum = zeros(dlp, max(lengths));
bden = zeros(dlp, max(lengths));
for trial = 1 : num_trials
	sumdlogPi = zeros(dlp,1);
    num_steps = size(data(trial).a,2);

	for step = 1 : num_steps
		sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
		rew = gamma^(step - 1) * data(trial).r(robj,step);
		sumdlogPi2 = sumdlogPi .* sumdlogPi;
		bnum(:,step) = bnum(:, step) + sumdlogPi2 * rew;
		bden(:,step) = bden(:, step) + sumdlogPi2; 
	end
end
b = bnum ./ bden;
b(isnan(b)) = 0; % When 0 / 0

% Compute gradient
for trial = 1 : num_trials
	sumdlogPi = zeros(dlp,1);
	for step = 1 : size(data(trial).a,2)
        sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        rew = gamma^(step-1) * data(trial).r(robj,step);
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dlp, 1) * rew - b(:,step));
        totstep = totstep + 1;
	end
end

if gamma == 1
    dJdtheta = dJdtheta / totstep;
else
    dJdtheta = dJdtheta / num_trials;
end

if nargin >= 5
    T = eye(length(dJdtheta)); % trasformation in Euclidean space
    lambda = sqrt(dJdtheta' * T * dJdtheta / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
end
