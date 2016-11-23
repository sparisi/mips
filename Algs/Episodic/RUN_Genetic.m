%% Settings
max_pop_size = 20;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
mutate = @(theta) Genetic_Solver.gaussianMutation(theta,policy_high.Sigma,0.2);
if makeDet, policy = policy.makeDeterministic; end

fitness_single = @(J) -J(robj,:); % Since population sorting is done in ascending order
fitness_population = @(J) max(J(robj,:));

solver = Genetic_Solver(elitism, mutation, crossover, mutate, ...
    max_pop_size, fitness_single, n_params);

% Initial population
current_population = policy.empty(max_pop_size,0);
Theta = policy_high.drawAction(max_pop_size);
for i = 1 : max_pop_size
    current_population(i) = policy.update(Theta(:,i));
end

% Evaluate the population
current_J = evaluate_policies(mdp, episodes_learn, steps_learn, current_population);

iter = 1;

%% Learning
while true

    dim(iter) = numel(current_population);
    current_fitness = fitness_population(current_J);
    fitness_history(iter) = current_fitness;
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, current_fitness, numel(current_population) );
    
    offspring = solver.mate(current_population);
    offspring_J = evaluate_policies(mdp, episodes_learn, steps_learn, offspring);
    [current_population, current_J] = solver.evolve( ...
        current_population, current_J, offspring, offspring_J );
    
    iter = iter + 1;

end

%% See best learned policy
[value, idx] = max(current_J(robj,:));
best_policy = current_population(idx);
show_simulation(mdp, best_policy, 1000, 0.1);