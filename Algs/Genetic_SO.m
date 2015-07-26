clear all
domain = 'dam';
[n_obj, pol] = feval([domain '_settings']);
[~, ~, utopia, antiutopia] = feval([domain '_moref'],0);
dim_theta = length(pol.theta);
robj = 2;

% Settings
makeDet = 0; % evaluate only deterministic policies?
max_pop_size = 10;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
% mutate = @Genetic_Solver.myMutation;
mutate = @(theta)Genetic_Solver.gaussianMutation(theta,50*ones(1,dim_theta),0.2);

solver = Genetic_Solver(elitism, mutation, crossover, mutate, max_pop_size);

% Initial population
current_population = pol.empty(max_pop_size,0);

dist = gaussian_diag_constant(dim_theta, pol.theta, pol.theta + ones(dim_theta,1));
for i = 1 : max_pop_size
    new_pol = pol;
    new_pol.theta = dist.drawAction;
    current_population(i) = new_pol;
end

% Evaluate the population
current_J = evaluate_policies(current_population, domain, makeDet);
current_J = current_J(:,robj);

iter = 0;
dim = max_pop_size;

%% Learning
while true

    iter = iter + 1;
    dim = [dim; numel(current_population)]; % Save the new size of the population

    % Evaluate the population
    current_fitness = max(current_J(robj));
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, current_fitness, numel(current_population) );
    
    % Evolve
    offspring = solver.getOffspring(current_population);
    offspring_J = evaluate_policies(offspring, domain, makeDet);
    offspring_J = offspring_J(:,robj);
    
    [current_population, current_J] = solver.getNewPopulation( ...
        current_population, current_J, offspring, offspring_J );
    
end
