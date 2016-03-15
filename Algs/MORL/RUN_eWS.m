% Episodic RL algorithms with a weighted scalarization of the returns for 
% multi-objective problems. See METRIC functions for details about the 
% scalarizations.

N = 20;
N_MAX = N * 10;
if makeDet, policy = policy.makeDeterministic; end
solver = REPS_Solver(0.9);
solver = NES_Solver(0.1);

verboseOut = true;
utopia = mdp.utopia;
antiutopia = mdp.antiutopia;

metric = @(r,w)metric_ws(r,w); % scalarization function
step = 10; % density of the weights
W = convexcomb(dreward, step);
ndir = size(W,1);

front_pol = cell(ndir,1);
iter = 0;
maxIter = 120;


%% Learning
for i = 1 : ndir
    
    current_pol = policy_high;
    J = zeros(dreward,N_MAX);
    Theta = zeros(current_pol.daction,N_MAX);
    
    current_iter = 1;
    str_dir = strtrim(sprintf('%.3f, ', W(i,:)));
    str_dir(end) = [];

    %% Single objective optimization
    while true
        [data, avgRew] = collect_episodes(mdp, N, steps_learn, current_pol, policy);
        
        % First, fill the pool to maintain the samples distribution
        if current_iter == 1
            J = repmat(min(data.J,[],2),1,N_MAX);
            Theta = current_pol.drawAction(N_MAX);
        end
        
        % Enqueue the new samples and remove the old ones
        J = [data.J, J(:, 1:N_MAX-N)];
        Theta = [data.Theta, Theta(:, 1:N_MAX-N)];
        
        % Scalarized return
        Jw = metric(J,W(i,:)');

        % Perform an update step
        [current_pol, div] = solver.step(Jw, Theta, current_pol);

        % Print info
        if verboseOut
            str_obj = strtrim(sprintf('%.4f, ', avgRew));
            str_obj(end) = [];
            fprintf('[ %s ] || Iter %d ) Dev: %.4f, \t J = [ %s ] \n', ...
                str_dir, current_iter, div, str_obj)
        end

        % Ending condition
        if div < 0.05 || current_iter > maxIter, break, end
        
        current_iter = current_iter + 1;
        iter = iter + 1;
    end
    
    front_pol{i} = current_pol;
    
end


%% Eval
front_pol = vertcat(front_pol{:});
fr = evaluate_policies_high(mdp, episodes_eval, steps_eval, policy, front_pol);
[f, p] = pareto(fr', front_pol);
fig = mdp.plotfront(mdp.truefront);
fig = mdp.plotfront(f,fig);
