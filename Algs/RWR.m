clear all
domain = 'puddlec';
robj = 1;
[~, policy, episodes, steps, gamma] = feval([domain '_settings']);
mdp_vars = feval([domain '_mdpvariables']);
iter = 0;

epsilon = 0.9;

%% Learning
while true
    
    iter = iter + 1;
    [data, J, S] = collect_samples(domain, episodes, steps, policy);

    count = 1;
    for trial = 1 : max(size(data));
        for step = 1 : size(data(trial).a,2)
            Action(count,:) = data(trial).a(:,step);
            Phi(count,:) = policy.basis(data(trial).s(:,step));
            R(count) = gamma^(step - 1) * sum(data(trial).r(robj,step:end));
            count = count + 1;
        end
    end
    
    weights = (R - min(R)) / (max(R) - min(R)); % simple normalization in [0,1]
%     beta = 0.1; weights = exp(beta*R);
    
    str_obj = strtrim(sprintf('%.4f, ', J));
    str_obj(end) = [];
    fprintf('%d ) Entropy: %.2f, J: [ %s ]\n', iter, S, str_obj)
    
    policy = policy.weightedMLUpdate(weights, Action, Phi);
    
end