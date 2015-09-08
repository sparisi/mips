clear all
domain = 'deep';
robj = 1;
[~, policy, episodes, steps, gamma] = feval([domain '_settings']);
mdp_vars = feval([domain '_mdpvariables']);
iter = 0;

epsilon = 0.9;
% phiV = @(varargin)basis_poly(1,mdp_vars.nvar_state,1,varargin{:});
% phiV = @(varargin)basis_krbf(5,[-30,0; -30,0],varargin{:});
phiV = policy.basis;
solver = CREPS_Solver(epsilon,policy,phiV);

%% Learning
while true
    
    iter = iter + 1;
    [data, J, S] = collect_samples(domain, episodes, steps, policy);

    count = 1;
    for trial = 1 : max(size(data));
        for step = 1 : size(data(trial).a,2)
            Action(count,:) = data(trial).a(:,step);
            PhiP(count,:) = policy.basis(data(trial).s(:,step));
            PhiVFun(count,:) = phiV(data(trial).s(:,step));
            R(count) = gamma^(step - 1) * (data(trial).r(robj,step));
            count = count + 1;
        end
    end
    
    % RWR uses simple weights
%     weights = (R - min(R)) / (max(R) - min(R)); % simple normalization in [0,1]
%     beta = 0.1; weights = exp(beta*R);
    
    % REPS finds weights to bound the KL divergence
    [weights, divKL] = solver.optimize(R, PhiVFun);

    str_obj = strtrim(sprintf('%.4f, ', J));
    str_obj(end) = [];
    fprintf('%d ) Entropy: %.2f, Div KL: %.3f, J: [ %s ]\n', iter, S, divKL, str_obj)
    
    if divKL < 0.001
        break
    end
    
    policy = policy.weightedMLUpdate(weights, Action, PhiP);
    
end