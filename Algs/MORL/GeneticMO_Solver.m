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
                mutate, max_size, fitness, n_params, min_size)
            obj = obj@Genetic_Solver(elitism, mutation, crossover, ...
                mutate, max_size, fitness, n_params);
            if nargin == 8, obj.min_size = min_size; end
        end

        %% INDIVIDUALS SELECTION
        function [new_population, new_J] = evolve(obj, population, J, offspring, J_offspring)
            % Generate the population as usual
            [new_population, new_J] = evolve@Genetic_Solver(obj, population, J, offspring, J_offspring);
            all_pop = [population, offspring];
            all_J = [J, J_offspring];
            
            % Filter dominated solutions
            [new_J, new_population, idx_pareto] = pareto(new_J', new_population);
            new_J = new_J';
            
            % If the population is too small, also include random
            % individuals from the previous generation
            n = numel(new_population);
            if n < obj.min_size
                idx = 1 : numel(all_pop);
                idx = idx(~ismember(idx, idx_pareto));
                idx = idx(randperm(length(idx), obj.min_size-n));
                new_J = [new_J, all_J(:,idx)];
                new_population = [new_population, all_pop(:,idx)];
            end                
        end
        
    end
    
end
