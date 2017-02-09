classdef GeneticMO_Solver < Genetic_Solver
% Extension of GENETIC_SOLVER for multi-objective problems. At each 
% evolution, dominated solutions are filtered out. If the population size 
% decreases below a threshold, also dominated solutions are kept.
    
    properties
        min_size = 3; % Minimum population size
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = GeneticMO_Solver(elitism, mutation, crossover, ...
                mutate, max_size, fitness, min_size)
            obj = obj@Genetic_Solver(elitism, mutation, crossover, ...
                mutate, max_size, fitness);
            if nargin == 8, obj.min_size = min_size; end
        end

        %% INDIVIDUALS SELECTION
        function [new_population, new_population_value] = evolve(obj, ...
                population, population_value, offspring, offspring_value)
            % Generate the population as usual
            [new_population, new_population_value] = evolve@Genetic_Solver(obj, ...
                population, population_value, offspring, offspring_value);
            all_pop = [population, offspring];
            all_values = [population_value, offspring_value];
            
            % Filter dominated solutions
            [new_population_value, new_population, idx_pareto] = ...
                pareto(new_population_value', new_population');
            new_population_value = new_population_value';
            new_population = new_population';
            
            % If the population is too small, also include random
            % individuals from the previous generation
            n = size(new_population,2);
            if n < obj.min_size
                idx = 1 : size(all_pop,2);
                idx = idx(~ismember(idx, idx_pareto));
                idx = idx(randperm(length(idx), obj.min_size-n));
                new_population_value = [new_population_value, all_values(:,idx)];
                new_population = [new_population, all_pop(:,idx)];
            end                
        end
        
    end
    
end
