% Radial Algorithm.
% 
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

simplexstep = 10; % Density of the directions in the simplex
tolerance = 0.001; % Tolerance for the norm of the gradient
maxIter = 200; % Max number of policy gradient steps in the same direction
minS = 0.1; % Min entropy of the policy (stopping condition)
lrate = 1;
gradient = @eNACbase;
verboseOut = false;

% Generate all combinations of weights for the directions in the simplex
W = convexcomb(dreward, simplexstep);
ndir = size(W,1);
front_pol = cell(ndir,0);

%% Initial solution
[ds, J_init] = collect_samples(mdp, episodes_learn, steps_learn, policy);
[grads_init, stepsize] = gradient(policy,ds,gamma,lrate);
grads_init = bsxfun(@times, grads_init, stepsize); % Apply stepsize
iter = 1;

%% For all the directions in the simplex
parfor k = 1 : ndir
    fixedLambda = W(k,:)';
    newDir = grads_init * fixedLambda;
    curr_pol = policy.update(policy.theta + lrate * newDir);
    str_dir = strtrim(sprintf('%.3f, ', fixedLambda));
    str_dir(end) = [];
    
    % Policy gradient learning in one fixed direction
    iter_dir = 1;
    while true
        [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, curr_pol);
        S = curr_pol.entropy(horzcat(ds.s));
        
        [grads, stepsize] = gradient(policy,ds,gamma,lrate);
        grads = bsxfun(@times, grads, stepsize); % Apply stepsize
        normgrads = matrixnorms(grads,2);
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
            fprintf('Cannot proceed any further in this direction!\n');
            front_pol{k} = curr_pol;
            break
        end
        
        % For the min-norm Pareto-ascent direction, always use the normalized gradients
        dirPareto = paretoDirection(gradsn);
        devPareto = norm(dirPareto);
        
        if devPareto < tolerance
            fprintf('Pareto front reached!\n');
            front_pol{k} = curr_pol;
            break
        end
        
        if S < minS
            fprintf('Deterministic policy found!\n');
            front_pol{k} = curr_pol;
            break
        end
        
        if iter_dir > maxIter
            fprintf('Iteration limit reached!\n');
            front_pol{k} = curr_pol;
            break
        end
        
        iter_dir = iter_dir + 1;
        iter = iter + 1;
        curr_pol = curr_pol.update(curr_pol.theta + dir);
    end
end

%% Eval
front_pol = vertcat(front_pol{:});
fr = evaluate_policies(mdp, episodes_eval, steps_eval, front_pol);
[f, p] = pareto(fr', front_pol);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
