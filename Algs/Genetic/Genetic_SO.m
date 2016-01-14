%% Settings
max_pop_size = 10;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
% mutate = @Genetic_Solver.myMutation;
mutate = @(theta)Genetic_Solver.gaussianMutation(theta,policy_high.Sigma,0.2);

solver = Genetic_Solver(elitism, mutation, crossover, mutate, max_pop_size);

% Initial population
current_population = policy.empty(max_pop_size,0);
Theta = policy_high.drawAction(max_pop_size);
for i = 1 : max_pop_size
    current_population(i) = policy.update(Theta(:,i));
end

% Evaluate the population
current_J = evaluate_policies(mdp, episodes_learn, steps_learn, current_population);
current_J = current_J(robj,:)';

dim = max_pop_size;
iter = 1;

%% Learning
while true

    dim(iter) = numel(current_population); % Save the new size of the population

    % Evaluate the population
    current_fitness = max(current_J);
    fitness_history(iter) = current_fitness;
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, current_fitness, numel(current_population) );
    
    % Evolve
    offspring = solver.getOffspring(current_population);
    offspring_J = evaluate_policies(mdp, episodes_learn, steps_learn, offspring);
    offspring_J = offspring_J(robj,:)'; % Genetic algs want samples in rows
    
    [current_population, current_J] = solver.getNewPopulation( ...
        current_population, current_J, offspring, offspring_J );
    
    iter = iter + 1;

end
