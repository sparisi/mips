classdef Genetic_Solver < handle
% Generic implementation of a genetic algorithm. Individuals are vectors of 
% parameters of size [D x N], where N is the number of individuals.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        elitism   % elitism percentage
        mutation  % mutation chance
        crossover % crossover function
        mutate    % mutation function
        max_size  % max population size
        fitness   % function to assess the fitness of single individuals
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = Genetic_Solver(elitism, mutation, crossover, ...
                mutate, max_size, fitness)
            obj.elitism = elitism;
            obj.mutation = mutation;
            obj.crossover = crossover;
            obj.mutate = mutate;
            obj.max_size = max_size;
            obj.fitness = fitness;
        end
        
        %% INDIVIDUALS EVOLUTION
        function offspring = mate(obj, population)
            n = size(population,2);
            
            % Get random pairs of individuals
            k = randperm(n/2*(n-1),n);
            q = floor(sqrt(8*(k-1) + 1)/2 + 3/2);
            p = k - (q-1).*(q-2)/2;
            parents_idx = sortrows([p;q]',[2 1]);
            
            % Select pairs to be mutated
            mutated = rand(1,n) < obj.mutation;
            
            sample1 = population(:, parents_idx(:,1));
            sample2 = population(:, parents_idx(:,2));
            
            % Perform crossover and mutation
            offspring = obj.crossover(sample1, sample2);
            offspring(:, mutated) = obj.mutate(offspring(:, mutated));
        end

        %% INDIVIDUALS SELECTION
        function [new_population, new_population_value] = evolve(obj, ...
                population, population_value, offspring, offspring_value)
            population_size = size(population,2);
            offspring_size = size(offspring,2);
            assert(population_size == size(population_value,2) && ...
                offspring_size == size(offspring_value,2), ...
                'Number of individuals and fitnesses not consistent.')
            
            % If the max size has not been reached, keep them all
            if size(population,2) + size(offspring,2) <= obj.max_size
                new_population = [population, offspring];
                new_population_value = [population_value, offspring_value];
                return
            end
            
            % Otherwise take the elites of the current population...
            n_elites = ceil( population_size * obj.elitism );
            current_fitness = obj.fitness(population_value);
            [~, indices] = sort(current_fitness);
            sorted_population = population(:, indices);
            sorted_value = population_value(:, indices);
            elites = sorted_population(:, 1:n_elites);
            elites_value = sorted_value(:, 1:n_elites);
            
            % ...remove them from the remaining individuals...
            sorted_population = sorted_population(:, n_elites+1:end);
            sorted_value = sorted_value(:, n_elites+1:end);
            
            % ...merge the remaining ones with the offspring, sort them and 
            % take the best ones
            n_remaining = obj.max_size - n_elites;
            remaining = [sorted_population, offspring];
            remaining_values = [sorted_value, offspring_value];
            remaining_fitness = obj.fitness(remaining_values);
            [~, indices] = sort(remaining_fitness);
            sorted_remaining = remaining(:, indices);
            sorted_value = remaining_values(:, indices);
            
            new_population = [elites, sorted_remaining(:, 1:n_remaining)];
            new_population_value = [elites_value, sorted_value(:, 1:n_remaining)];
        end
        
    end
    
    methods(Static)
        
        %% CROSSOVERS
        function new_sample = uniformCrossover(sample1, sample2)
            idx = rand(size(sample1)) < 0.5;
            new_sample = zeros(size(sample1));
            new_sample(idx) = sample1(idx);
            new_sample(~idx) = sample2(~idx);
        end

        %% MUTATORS
        function sample = gaussianMutation(sample, noise, chance)
        % Adds Gaussian noise to random chromosomes.
            idx = rand(size(sample)) < chance;
            wNoise = mymvnrnd(zeros(size(sample,1),1), noise, size(sample,2));
            sample(idx) = sample(idx) + wNoise(idx);
        end
        
        function sample = integerMutation(sample, minvalue, maxvalue, chance)
            idx = rand(size(sample)) < chance;
            sample(idx) = randi([minvalue,maxvalue],size(sample(idx)));
        end
        
    end
    
end
