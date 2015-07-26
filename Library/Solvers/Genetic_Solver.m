classdef Genetic_Solver < handle
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        elitism;   % elitism percentage
        mutation;  % mutation chance
        crossover; % crossover function
        mutate;    % mutation function
        max_size;  % max population size
    end

    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = Genetic_Solver(elitism, mutation, crossover, mutate, max_size)
            obj.elitism = elitism;
            obj.mutation = mutation;
            obj.crossover = crossover;
            obj.mutate = mutate;
            obj.max_size = max_size;
        end
        
        %% INDIVIDUALS EVOLUTION
        function offspring = getOffspring ( obj, current_population )
            population_size = numel(current_population);
            offspring = current_population(1).empty(population_size,0);

            for i = 1 : population_size
                % Get 2 parents
                idx1 = randi(population_size,1);
                idx2 = randi(population_size,1);
                theta1 = current_population(idx1).theta;
                theta2 = current_population(idx2).theta;
                
                % Do the crossover
                new_theta = obj.crossover(theta1,theta2);
                
                % Mutate
                if rand < obj.mutation
                    new_theta = obj.mutate(new_theta);
                end
                
                % Add the child to the offspring
                child = current_population(idx1);
                child.theta = new_theta;
                offspring(i) = child;
            end
        end

        %% INDIVIDUALS SELECTION
        function [new_population, new_J] = getNewPopulation( obj, ...
                current_population, current_J, offspring, offspring_J)
            population_size = numel(current_population);
            
            % If the max size has not been reached, keep them all
            all_pop = [current_population, offspring];
            all_J = [current_J; offspring_J];
            % In MO problems, filter the population and take only non-dominated solutions
            if size(all_J,2) > 1
                [all_J, all_pop] = pareto(all_J, all_pop);
            end
            if numel(all_pop) < obj.max_size
                new_population = all_pop;
                new_J = all_J;
                return
            end
            
            % Otherwise take the elites of the current population...
            n_elites = ceil( population_size * obj.elitism );
            current_fitness = obj.getFitness(current_J);
            [~, indices] = sort(current_fitness,1,'ascend');
            sorted_population = current_population(indices);
            sorted_J = current_J(indices,:);
            elites = sorted_population(1:n_elites);
            elites_J = sorted_J(1:n_elites,:);
            
            % ...remove them from the remaining individuals...
            sorted_population = sorted_population(n_elites+1:end);
            sorted_J = sorted_J(n_elites+1:end,:);
            
            % ...merge the remaining ones with the offspring, sort them and 
            % take the best ones
            n_remaining = population_size - n_elites;
            remaining = [sorted_population, offspring];
            remaining_J = [sorted_J; offspring_J];
            remaining_fitness = obj.getFitness(remaining_J);
            [~, indices] = sort(remaining_fitness,1,'ascend');
            sorted_remaining = remaining(indices);
            sorted_J = remaining_J(indices,:);
            
            new_population = [elites, sorted_remaining(1:n_remaining)];
            new_J = [elites_J; sorted_J(1:n_remaining,:)];
        end
        
        %% SIMPLE SINGLE-OBJECTIVE FITNESS. MULTI-OBJECTIVE ALGORITHMS 
        % OVERRIDE IT
        function J = getFitness(obj, J)
            J = -J; % Since population sorting is done in ascending order
        end
        
    end
    
    methods(Static)
        
        %% CROSSOVERS
        function new_theta = uniformCrossover(theta1, theta2)
            new_theta = zeros(size(theta1));
            for j = 1 : size(theta1,1)
                if rand < 0.5
                    new_theta(j) = theta1(j);
                else
                    new_theta(j) = theta2(j);
                end
            end
        end

        %% MUTATORS
        function theta = myMutation(theta)
            idx = randi(length(theta),1);
            
            if rand < 0.5 % Add or subtract the mean value
                mutation_value = mean(theta);
                if rand < 0.5
                    mutation_value = - mutation_value;
                end
                theta(idx) = theta(idx) + mutation_value;
            else
                mutation_value = 2; % Halve or double
                if rand < 0.5
                    mutation_value = 1 / mutation_value;
                end
                theta(idx) = theta(idx) * mutation_value;
            end
        end
        
        function theta = gaussianMutation(theta, noise, chance)
        % Adds Gaussian noise to random chromosomes.
            n_noise = 0;
            while n_noise <= 0 % Do at least one perturbation
                randIdx = rand(size(theta)) < chance;
                n_noise = sum(randIdx);
            end
            wNoise = mvnrnd(zeros(n_noise,1), diag(noise(randIdx)));
            theta(randIdx) = theta(randIdx) + wNoise';
        end
        
    end
    
end
