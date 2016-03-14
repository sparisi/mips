% Episodic Pareto-Following Algorithm.
% 
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

tolerance_step = 0.1; % Tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.25; % The same, but during the correction step
minS = 1.5; % Min entropy of the policy (stopping condition)
lrate_step = 0.5; % Lrate during optimization step
lrate_corr = 5; % Lrate during correction step
maxIter = 200; % Max iterations per optimization
maxCorrection = 100; % Max iterations during the correction
randfactor = 1.5; % Randomization after single-objective optimization
alg = @NESbase;
solver = ePFA_Solver(lrate_step,lrate_corr,alg);
iter = 1;
verboseOut = true;

%% Learn the last objective
while true
    [J, Theta] = collect_episodes(mdp, episodes_learn, steps_learn, policy_high, policy);
    S = policy_high.entropy();
    [policy_high, gnorm] = solver.optimization_step(J, Theta, policy_high, dreward);
    if verboseOut, fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s\n', ...
            iter, S, gnorm, num2str(mean(J,2)','%.4f, ')); end
    if gnorm < tolerance_step || S < minS || iter > maxIter, break, end
    iter = iter + 1;
end

front_J = mean(J,2);
front_pol = policy_high;
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
            [current_J, current_Theta] = collect_episodes(mdp, episodes_learn, steps_learn, current_pol, policy);
            S = current_pol.entropy();
            [current_pol, gnorm] = solver.optimization_step(current_J, current_Theta, current_pol, obj);

            if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Optimization\n', ...
                    i, num_policy, obj, S, gnorm, num2str(mean(current_J,2)','%.4f, ')); end

            if gnorm < tolerance_step || iter_opt > maxIter || S < minS, break, end
            inter_pol = [inter_pol, current_pol]; % Save intermediate solution
            inter_J = [inter_J, mean(current_J,2)];
            
            iter_opt = iter_opt + 1;
            iter = iter + 1;
            
            if iter * episodes_learn > 1e6, break; end % Budget limit

            iter_corr = 1; % Iter counter for the correction step
            while true % Correction step
                [current_J, current_Theta] = collect_episodes(mdp, episodes_learn, steps_learn, current_pol, policy);
                S = current_pol.entropy();
                [current_pol, gnorm] = solver.correction_step(current_J, current_Theta, current_pol);

                if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Correction\n', ...
                        i, num_policy, obj, S, gnorm, num2str(mean(current_J,2)','%.4f, ')); end

                if gnorm < tolerance_corr || iter_corr > maxCorrection || S < minS, break, end
                inter_pol = [inter_pol, current_pol]; % Save intermediate solution
                inter_J = [inter_J, mean(current_J,2)];
                iter_corr = iter_corr + 1;
                iter = iter + 1;
            end
        end
    end
end

%% Eval
fr = evaluate_policies_high(mdp, episodes_eval, steps_eval, policy, [front_pol, inter_pol]);
[f, p] = pareto(fr', [front_pol, inter_pol]);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
