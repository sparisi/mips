%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: J Peters, S Schaal
% Reinforcement Learning by Reward-weighted Regression for Operational 
% Space Control (2007)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
domain = 'puddlec';
robj = 1;
[~, policy, episodes, steps, gamma] = feval([domain '_settings']);
mdp_vars = feval([domain '_mdpvariables']);
iter = 0;
NMAX = 10000;
count = 1;

%% Learning
while true
    
    iter = iter + 1;
    [data, J, S] = collect_samples(domain, episodes, steps, policy);

    for trial = 1 : max(size(data));
        maxsteps = size(data(trial).a,2);
        for step = 1 : maxsteps
            Action(count,:) = data(trial).a(:,step);
            Phi(count,:) = policy.basis(data(trial).s(:,step));
            gammav = gamma.^((step:maxsteps)-1);
            R(count) = sum(gammav .* (data(trial).r(robj,step:end)));
            count = count + 1;
            if count > NMAX
                count = 1;
            end
        end
    end
    
    weights = (R - min(R)) / (max(R) - min(R)); % simple normalization in [0,1]
%     beta = 0.1; weights = exp(beta*R);
    
    str_obj = strtrim(sprintf('%.4f, ', J));
    str_obj(end) = [];
    fprintf('%d ) Entropy: %.2f, J: [ %s ]\n', iter, S, str_obj)
    
    policy = policy.weightedMLUpdate(weights, Action, Phi);
    
end