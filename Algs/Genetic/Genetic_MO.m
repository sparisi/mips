%% Settings
max_pop_size = 20;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
% mutate = @Genetic_Solver.myMutation;
mutate = @(theta)Genetic_Solver.gaussianMutation(theta,policy_high.Sigma,0.2);

if mdp.dreward == 2
    fitness = @(f)hypervolume2d(f,mdp.antiutopia,mdp.utopia);
else
    fitness = @(f)hypervolume(f,mdp.antiutopia,mdp.utopia,1e6);
end

solver = SMSEMOA_Solver(elitism, mutation, crossover, mutate, max_pop_size, fitness);
% solver = NSGA2_Solver(elitism, mutation, crossover, mutate, max_pop_size);

% Initial population
current_population = policy.empty(max_pop_size,0);
Theta = policy_high.drawAction(max_pop_size);
for i = 1 : max_pop_size
    current_population(i) = policy.update(Theta(:,i));
end

%% Evaluate the population and take only the non-dominated solutions
current_J = evaluate_policies(mdp, episodes_learn, steps_learn, current_population);
[current_J, current_population] = pareto(current_J', current_population);

dim = max_pop_size;
iter = 1;

%% Learning
while true

    dim(iter) = numel(current_population); % Save the new size of the population

    % Evaluate the population
    current_fitness = fitness(current_J);
    fitness_history(iter) = current_fitness;
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, current_fitness, numel(current_population) );
    
    % Evolve
    offspring = solver.getOffspring(current_population);
    offspring_J = evaluate_policies(mdp, episodes_learn, steps_learn, offspring);
    offspring_J = offspring_J';
    
    [current_population, current_J] = solver.getNewPopulation( ...
        current_population, current_J, offspring, offspring_J );
    
    iter = iter + 1;

end

%% Evaluate and plot
J = evaluate_policies(mdp, episodes_eval, steps_eval, current_population);
[f, p] = pareto(J', current_population);
mdp.plotfront(f)