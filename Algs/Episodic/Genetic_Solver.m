classdef Genetic_Solver < handle
% Generic implementation of a genetic algorithm. Individuals are instances 
% of the POLICY class. The evolution is performed on their parameters THETA.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        elitism   % elitism percentage
        mutation  % mutation chance
        crossover % crossover function
        mutate    % mutation function
        max_size  % max population size
        fitness   % function to assess the fitness of single individuals
        n_params  % number of policy parameters (can change if the policy is deterministic)
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = Genetic_Solver(elitism, mutation, crossover, ...
                mutate, max_size, fitness, n_params)
            obj.elitism = elitism;
            obj.mutation = mutation;
            obj.crossover = crossover;
            obj.mutate = mutate;
            obj.max_size = max_size;
            obj.fitness = fitness;
            obj.n_params = n_params;
        end
        
        %% INDIVIDUALS EVOLUTION
        function offspring = mate(obj, population)
            n = numel(population);
            offspring = population(1).empty(n,0);
            
            % Get random pairs of individuals
            k = randperm(n/2*(n-1),n);
            q = floor(sqrt(8*(k-1) + 1)/2 + 3/2);
            p = k - (q-1).*(q-2)/2;
            parents_idx = sortrows([p;q]',[2 1]);
            
            % See which pair will go under mutation
            mutated = rand(1,n) < obj.mutation;
            
            all_theta = [population.theta];
            all_theta = all_theta(1:obj.n_params, :);
            theta1 = all_theta(:, parents_idx(:,1));
            theta2 = all_theta(:, parents_idx(:,2));
            
            % Perform crossover and mutation
            new_theta = obj.crossover(theta1, theta2);
            new_theta(:,mutated) = obj.mutate(new_theta(:, mutated));
            
            % Generate offspring
            for i = 1 : n
                offspring(i) = population(1).update(new_theta(:,i));
            end
        end

        %% INDIVIDUALS SELECTION
        function [new_population, new_J] = evolve(obj, population, J, offspring, J_offspring)
            population_size = numel(population);
            offspring_size = numel(offspring);
            J_size = size(J, 2);
            Joff_size = size(J_offspring, 2);
            assert(population_size == J_size && offspring_size == Joff_size, ...
                'Number of individuals and fitnesses not consistent.')
            
            all_pop = [population, offspring];
            all_J = [J, J_offspring];
            
            % If the max size has not been reached, keep them all
            if numel(all_pop) <= obj.max_size
                new_population = all_pop;
                new_J = all_J;
                return
            end
            
            % Otherwise take the elites of the current population...
            n_elites = ceil( population_size * obj.elitism );
            current_fitness = obj.fitness(J);
            [~, indices] = sortrows(current_fitness);
            sorted_population = population(indices);
            J_sorted = J(:, indices);
            elites = sorted_population(1:n_elites);
            J_elites = J_sorted(:, 1:n_elites);
            
            % ...remove them from the remaining individuals...
            sorted_population = sorted_population(n_elites+1: end);
            J_sorted = J_sorted(:, n_elites+1:end);
            
            % ...merge the remaining ones with the offspring, sort them and 
            % take the best ones
            n_remaining = obj.max_size - n_elites;
            remaining = [sorted_population, offspring];
            remaining_J = [J_sorted, J_offspring];
            remaining_fitness = obj.fitness(remaining_J);
            [~, indices] = sortrows(remaining_fitness);
            sorted_remaining = remaining(indices);
            J_sorted = remaining_J(:, indices);
            
            new_population = [elites, sorted_remaining(1:n_remaining)];
            new_J = [J_elites, J_sorted(:, 1:n_remaining)];
        end
        
    end
    
    methods(Static)
        
        %% CROSSOVERS
        function new_theta = uniformCrossover(theta1, theta2)
            idx = rand(size(theta1)) < 0.5;
            new_theta = zeros(size(theta1));
            new_theta(idx) = theta1(idx);
            new_theta(~idx) = theta2(~idx);
        end

        %% MUTATORS
        function theta = gaussianMutation(theta, noise, chance)
        % Adds Gaussian noise to random chromosomes.
            idx = rand(size(theta)) < chance;
            wNoise = mymvnrnd(zeros(size(theta,1),1), noise, size(theta,2));
            theta(idx) = theta(idx) + wNoise(idx);
        end
        
    end
    
end
