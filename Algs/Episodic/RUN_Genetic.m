%% Settings
if makeDet, policy = policy.makeDeterministic; end

max_pop_size = 20;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
mutate = @(theta) Genetic_Solver.gaussianMutation(theta,policy_high.Sigma,0.2);

fitness_single = @(J) -J(robj,:); % Since population is sorted in ascending order
fitness_population = @(J) max(J(robj,:));

solver = Genetic_Solver(elitism, mutation, crossover, mutate, ...
    max_pop_size, fitness_single);

% Initial population
Theta = policy_high.drawAction(max_pop_size);
for i = 1 : max_pop_size
    Policies(i) = policy.update(Theta(:,i));
end

% Evaluate the population
J = evaluate_policies(mdp, episodes_learn, steps_learn, Policies);

iter = 1;

%% Learning
while true

    dim(iter) = size(Theta,2);
    fitness_history(iter) = fitness_population(J);
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, fitness_history(iter), dim(iter) );
    
    Theta_New = solver.mate(Theta);
    Policies = policy.empty(0,size(Theta_New,2));
    for i = 1 : size(Theta_New,2)
        Policies(i) = policy.update(Theta_New(:,i));
    end
    J_New = evaluate_policies(mdp, episodes_learn, steps_learn, Policies);
    [Theta, J] = solver.evolve( Theta, J, Theta_New, J_New );
    
    iter = iter + 1;

end

%% See best learned policy
[value, idx] = max(J(robj,:));
best_policy = Policies(idx);
show_simulation(mdp, best_policy, 1000, 0.01)