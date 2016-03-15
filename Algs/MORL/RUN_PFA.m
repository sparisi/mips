% Pareto-Following Algorithm.
% 
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)


%% Step-based settings
tolerance_step = 0.05; % Tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.05; % The same, but during the correction step
minS = 0.1; % Min entropy of the policy (stopping condition)
lrate_step = 0.1; % Lrate during optimization step
lrate_corr = 0.1; % Lrate during correction step
maxIter = 500; % Max iterations per optimization
maxCorrection = 25; % Max iterations during the correction
randfactor = 10; % Randomization after single-objective optimization

target_policy = policy; % Learn the low-level policy
alg = @(policy, data, lrate) eNACbase(policy, data, gamma, lrate);
collect = @(policy) collect_samples(mdp, episodes_learn, steps_learn, policy);
calc_entropy = @(data, policy) policy.entropy(horzcat(data.s));
eval = @(policies) evaluate_policies(mdp, episodes_eval, steps_eval, policies);


%% Episodic settings
tolerance_step = 0.1; % Tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.25; % The same, but during the correction step
minS = 1.5; % Min entropy of the policy (stopping condition)
lrate_step = 0.5; % Lrate during optimization step
lrate_corr = 5; % Lrate during correction step
maxIter = 200; % Max iterations per optimization
maxCorrection = 100; % Max iterations during the correction
randfactor = 1.5; % Randomization after single-objective optimization

target_policy = policy_high; % Learn the high-level policy
alg = @(policy, data, lrate) NESbase(policy, data, lrate);
collect = @(policy_high) collect_episodes(mdp, episodes_learn, steps_learn, policy_high, policy);
calc_entropy = @(data, policy) policy.entropy(data.Theta);
eval = @(policies) evaluate_policies_high(mdp, episodes_eval, steps_eval, policy, policies);




%% Ready
solver = PFA_Solver(lrate_step, lrate_corr, alg);
iter = 1;
verboseOut = true;


%% Learn the last objective
while true
    [ds, J] = collect(target_policy);
    S = calc_entropy(ds, target_policy);
    [target_policy, gnorm] = solver.optimization_step(ds, target_policy, dreward);
    if verboseOut, fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s\n', ...
            iter, S, gnorm, num2str(J','%.4f, ')); end
    if gnorm < tolerance_step || S < minS || iter > maxIter, break, end
    iter = iter + 1;
end

front_J = J;
front_pol = target_policy;
inter_pol = [];
inter_J = [];


%% For all the remaining objectives
for obj = linspace(dreward-1, 1, dreward-1)
    % Filter suboptimal policies
    all_policies = [front_pol, inter_pol]; 
    all_J = [front_J, inter_J];
    [front_J, front_pol] = pareto(all_J', all_policies);
    front_J = front_J';
    
    % Keep intermediate solutions
    inter_pol = [];
    inter_J = [];
    
    num_policy = numel(front_pol);
    
    %% For all the Pareto-optimal solutions found so far
    for i = 1 : num_policy
        current_pol = front_pol(i);
        current_pol = current_pol.randomize(randfactor);

        iter_opt = 1; % Iter counter for the optimization step
        while true % Optimization step
            [ds, J] = collect(current_pol);
            S = calc_entropy(ds, current_pol);
            [current_pol, gnorm] = solver.optimization_step(ds, current_pol, obj);

            if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Optimization\n', ...
                    i, num_policy, obj, S, gnorm, num2str(J','%.4f, ')); end

            if gnorm < tolerance_step || iter_opt > maxIter || S < minS, break, end
            inter_pol = [inter_pol, current_pol]; % Save intermediate solution
            inter_J = [inter_J, J];
            
            iter_opt = iter_opt + 1;
            iter = iter + 1;
            
            if iter * episodes_learn > 1e6, break; end % Budget limit

            iter_corr = 1; % Iter counter for the correction step
            while true % Correction step
                [ds, J] = collect(current_pol);
                S = calc_entropy(ds, current_pol);
                [current_pol, gnorm] = solver.correction_step(ds, current_pol);
                
                if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Correction\n', ...
                        i, num_policy, obj, S, gnorm, num2str(J','%.4f, ')); end

                if gnorm < tolerance_corr || iter_corr > maxCorrection || S < minS, break, end
                inter_pol = [inter_pol, current_pol]; % Save intermediate solution
                inter_J = [inter_J, J];
                iter_corr = iter_corr + 1;
                iter = iter + 1;
            end
        end
    end
end

%% Eval
fr = eval([front_pol, inter_pol]);
[f, p] = pareto(fr', [front_pol, inter_pol]);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
