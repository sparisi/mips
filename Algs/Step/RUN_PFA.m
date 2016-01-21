% Pareto-Following Algorithm.
% 
% =========================================================================
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)

tolerance_step = 0.5; % Tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.5; % The same, but during the correction step
minS = 0.1; % Min entropy of the policy (stopping condition)
lrate_single = 2; % Lrate during single-objective optimization phase
lrate_step = 1; % Lrate during optimization step
lrate_corr = 0.5; % Lrate during correction step
maxIter = 500; % Max iterations per optimization
maxCorrection = 25; % Max iterations during the correction
randfactor = 1; % Randomization after single-objective optimization
alg = @eNACbase;
solver = PFA_Solver(lrate_single,lrate_step,lrate_corr,gamma,alg);
iter = 1;
verboseOut = false;

%% Learn the last objective
while true
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy(horzcat(ds.s));
    [policy, gnorm] = solver.optimization_step(ds, policy, dreward);
    if verboseOut, fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s\n', ...
            iter, S, gnorm, num2str(J','%.4f, ')); end
    if gnorm < tolerance_step || S < minS || iter > maxIter, break, end
    iter = iter + 1;
end

front_J = J;
front_pol = policy;
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
            [ds, current_J] = collect_samples(mdp, episodes_learn, steps_learn, current_pol);
            S = current_pol.entropy(horzcat(ds.s));
            [current_pol, gnorm] = solver.optimization_step(ds, current_pol, obj);

            if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Optimization\n', ...
                    i, num_policy, obj, S, gnorm, num2str(current_J','%.4f, ')); end

            if gnorm < tolerance_step || iter_opt > maxIter || S < minS, break, end
            inter_pol = [inter_pol, current_pol]; % Save intermediate solution
            inter_J = [inter_J, current_J];
            
            iter_opt = iter_opt + 1;
            iter = iter + 1;
            
            if iter * episodes_learn > 1e6, break; end % Budget limit

            iter_corr = 1; % Iter counter for the correction step
            while true % Correction step
                [ds, current_J] = collect_samples(mdp, episodes_learn, steps_learn, current_pol);
                S = current_pol.entropy(horzcat(ds.s));
                [current_pol, gnorm] = solver.correction_step(ds, current_pol);
                
                if verboseOut, fprintf('Policy %d/%d, Obj %d) Entropy: %.2f \tNorm: %.2e \tJ: %s *** Correction\n', ...
                        i, num_policy, obj, S, gnorm, num2str(current_J','%.4f, ')); end

                if gnorm < tolerance_corr || iter_corr > maxCorrection || S < minS, break, end
                inter_pol = [inter_pol, current_pol]; % Save intermediate solution
                inter_J = [inter_J, current_J];
                iter_corr = iter_corr + 1;
                iter = iter + 1;
            end
        end
    end
end

%% Eval
fr = evaluate_policies(mdp, episodes_eval, steps_eval, [front_pol, inter_pol]);
[f, p] = pareto(fr', [front_pol, inter_pol]);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
