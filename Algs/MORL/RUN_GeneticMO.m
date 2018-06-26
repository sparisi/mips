%% Settings
policy = policy.makeDeterministic; % Learn deterministic low-level policy

max_pop_size = 20;
min_pop_size = 10;
elitism = 0.1;
mutation = 0.8;
crossover = @Genetic_Solver.uniformCrossover;
mutate = @(theta) Genetic_Solver.gaussianMutation(theta, policy_high.Sigma, 0.2);

if mdp.dreward == 2
    hvf = @(J) hypervolume2d(J, mdp.antiutopia, mdp.utopia); % Hypervolume function handle
else
    hvf = @(J) hypervolume(J, mdp.antiutopia, mdp.utopia, 1e6);
end    
% fitness_single = @(J) nds(J'); % NSGA-II
fitness_single = @(J) -metric_hv(J',hvf); % SMS-EMOA
fitness_population = @(J) hvf(J'); % Hypervolume of population

solver = GeneticMO_Solver( elitism, mutation, crossover, mutate, ...
    max_pop_size, fitness_single, min_pop_size );

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
    
    fprintf( 'Iteration %d, Hypervolume: %.4f, Population Size: %d\n', ...
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

%% Evaluate and plot
Policies = policy.empty(0,size(Theta,2));
for i = 1 : size(Theta_New,2)
    Policies(i) = policy.update(Theta(:,i));
end
J = evaluate_policies(mdp, episodes_eval, steps_eval, Policies);
[f, p] = pareto(J', Policies');
mdp.plotfront(mdp.truefront,'o','DisplayName','True frontier');
hold all
mdp.plotfront(f,'+','DisplayName','Approximate frontier');
legend show