% Radial Algorithm.
% 
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

%% Step-based settings
simplexstep = 10; % Density of the directions in the simplex
tolerance = 0.001; % Tolerance for the norm of the gradient
maxIter = 100; % Max number of policy gradient steps in the same direction
minS = 0.1; % Min entropy of the policy (stopping condition)
lrate = 0.1;

target_policy = policy; % Learn the low-level policy
gradient = @(policy, data, lrate) eNACbase(policy, data, gamma, lrate);
collect = @(policy) collect_samples(mdp, episodes_learn, steps_learn, policy);
calc_entropy = @(data, policy) policy.entropy(horzcat(data.s));
eval = @(policies) evaluate_policies(mdp, episodes_eval, steps_eval, policies);

%% Episodic settings
simplexstep = 10; % Density of the directions in the simplex
tolerance = 0.1; % Tolerance for the norm of the gradient
maxIter = 200; % Max number of policy gradient steps in the same direction
minS = 1.5; % Min entropy of the policy (stopping condition)
lrate = 1;
if makeDet, policy = policy.makeDeterministic; end

target_policy = policy_high; % Learn the high-level policy
gradient = @(policy, data, lrate) NESbase(policy, data, lrate);
collect = @(policy_high) collect_episodes(mdp, episodes_learn, steps_learn, policy_high, policy);
calc_entropy = @(data, policy) policy.entropy(data.Theta);
eval = @(policies) evaluate_policies_high(mdp, episodes_eval, steps_eval, policy, policies);




%% Ready
verboseOut = true;

% Initial solution
[ds, J] = collect(target_policy);
[grads, stepsize] = gradient(target_policy, ds, lrate);
grads = bsxfun(@times, grads, stepsize); % Apply stepsize
iter = 1;

% Generate all combinations of weights for the directions in the simplex
W = convexcomb(dreward, simplexstep);
ndir = size(W,1);


%% For all the directions in the simplex
for k = 1 : ndir
    fixedLambda = W(k,:)';
    newDir = grads * fixedLambda;
    curr_pol = target_policy.update(target_policy.theta + lrate * newDir);
    str_dir = strtrim(sprintf('%.3f, ', fixedLambda));
    str_dir(end) = [];
    
    % Policy gradient learning in one fixed direction
    iter_dir = 1;
    while true
        [ds, J] = collect(curr_pol);
        S = calc_entropy(ds, curr_pol);
        
        [grads, stepsize] = gradient(curr_pol, ds, lrate);
        grads = bsxfun(@times, grads, stepsize); % Apply stepsize
        normgrads = max(matrixnorms(grads,2),1e-8);
        gradsn = bsxfun(@times,grads,1./normgrads); % Normalized gradients
        
        dir = grads * fixedLambda;
        dev = norm(dir);
        
        if verboseOut
            str_obj = strtrim(sprintf('%.4f, ', J));
            str_obj(end) = [];
            fprintf('[ %s ] || Iter %d ) Norm: %.4f, S: %.2f, J: [ %s ] \n', ...
                    str_dir, iter_dir, dev, S, str_obj);
        end
        
        if dev < tolerance
            fprintf('Cannot proceed any further in this direction!\n\n');
            front_pol(k) = curr_pol;
            break
        end
        
        % For the min-norm Pareto-ascent direction, always use the normalized gradients
        dirPareto = paretoDirection(gradsn);
        devPareto = norm(dirPareto);
        
        if devPareto < tolerance
            fprintf('Pareto front reached!\n\n');
            front_pol(k) = curr_pol;
            break
        end
        
        if S < minS
            fprintf('Deterministic policy found!\n\n');
            front_pol(k) = curr_pol;
            break
        end
        
        if iter_dir > maxIter
            fprintf('Iteration limit reached!\n\n');
            front_pol(k) = curr_pol;
            break
        end
        
        iter_dir = iter_dir + 1;
        iter = iter + 1;
        curr_pol = curr_pol.update(curr_pol.theta + dir);
    end
end

%% Eval
fr = eval(front_pol);
[f, p] = pareto(fr', front_pol);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
