clear all
domain = 'dam';
[n_obj, pol] = feval([domain '_settings']);
[~, ~, utopia, antiutopia] = feval([domain '_moref'],0);
dim_theta = length(pol.theta);

% Settings
makeDet = 0; % evaluate only deterministic policies?
max_pop_size = 150;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
% mutate = @Genetic_Solver.myMutation;
mutate = @(theta)Genetic_Solver.gaussianMutation(theta,50*ones(1,dim_theta),0.2);

% fitness = @(varargin)-eval_loss(varargin{:},domain);
% fitness = @(varargin)hypervolume(varargin{:},antiutopia,utopia,1e6);
fitness = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);

solver = SMSEMOA_Solver(elitism, mutation, fitness, crossover, mutate, max_pop_size);
% solver = NSGA2_Solver(elitism, mutation, crossover, mutate, max_pop_size);

% Initial population
current_population = pol.empty(max_pop_size,0);

dist = gaussian_diag_constant(dim_theta, pol.theta, pol.theta + ones(dim_theta,1));
for i = 1 : max_pop_size
    new_pol = pol;
    new_pol.theta = dist.drawAction;
    current_population(i) = new_pol;
end

% Evaluate the population and take only the non-dominated solutions
current_J = evaluate_policies(current_population, domain, makeDet);
[current_J, current_population] = pareto(current_J, current_population);

iter = 0;
dim = max_pop_size;

%% Learning
while true

    iter = iter + 1;
    dim = [dim; numel(current_population)]; % Save the new size of the population

    % Evaluate the population
    [front, front_pol] = pareto(current_J, current_population');
    current_fitness = fitness(front);
    
    fprintf( 'Iteration %d, Fitness: %.4f, Population Size: %d\n', ...
        iter, current_fitness, numel(current_population) );
    
    % Evolve
    offspring = solver.getOffspring(current_population);
    offspring_J = evaluate_policies(offspring, domain, makeDet);
    
    [current_population, current_J] = solver.getNewPopulation( ...
        current_population, current_J, offspring, offspring_J );
    
end


%% Plot
f = evaluate_policies(front_pol, domain, makeDet);
[f, p] = pareto(f, front_pol);

figure; hold all
if n_obj == 2
    plot(f(:,1),f(:,2),'g+')
end

if n_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
end

feval([domain '_moref'],1);
