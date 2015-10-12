%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, stepsize] = eNAC(policy, data, gamma, robj, lrate)

dlp = policy.dlogPidtheta();
F = zeros(dlp+1, dlp+1); % Fisher matrix
g = zeros(dlp+1, 1); % Vanilla gradient
num_trials = max(size(data));

parfor trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    R = 0;
    
    for step = 1 : size(data(trial).a,2)
		% Derivative of the logarithm of the policy in (s_t, a_t)
		dlogpidtheta = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));

		% Basis functions
 		phi = phi + [gamma^(step - 1) * dlogpidtheta; 0];
		
		% Discounted reward
		R = R + gamma^(step - 1) * (data(trial).r(robj,step));
    end
    
    F = F + phi * phi';
    g = g + phi * R;
end

rankF = rank(F);
if rankF == dlp + 1
	w = F \ g;
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, dlp+1);
	w = pinv(F) * g;
end

if nargin >= 5
    lambda = sqrt(g' * w / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 / (2 * lambda);
end

w = w(1:end-1);
