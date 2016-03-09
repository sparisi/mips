classdef SMSEMOA_Solver < Genetic_Solver
% S-Metric Selection Evolutionary Multi-Objective Algorithm.
%
% =========================================================================
% REFERENCE
% N Beume, B Naujoks, M Emmerich
% SMS-EMOA: Multiobjective selection based on dominated hypervolume (2007)
    
    properties(GetAccess = 'public', SetAccess = 'private')
        fitness
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = SMSEMOA_Solver(elitism, mutation, crossover, mutate, max_size, fitness)
            obj = obj@Genetic_Solver(elitism, mutation, crossover, mutate, max_size);
            obj.fitness = fitness;
        end
        
        function values = getFitness ( obj, J )
        % In SMS-EMOA, a solution is ranked higher if removing it from the 
        % population the hypervolume decreases.
            [uniqueJ, ~, idx] = unique(J,'rows');
            fitnessUnique = zeros(size(uniqueJ,1),1);
            for i = 1 : size(uniqueJ,1)
                front_tmp = uniqueJ;
                front_tmp(i,:) = []; % Remove the i-th element from the pool
                fitnessUnique(i) = obj.fitness(pareto(front_tmp)); % Evaluate the fitness
            end
            values = fitnessUnique(idx);
        end
        
    end

end
