%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: N Beume, B Naujoks, M Emmerich (2007)
% SMS-EMOA: Multiobjective selection based on dominated hypervolume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef SMSEMOA_Solver < Genetic_Solver
    
    % S-Metric Selection Evolutionary Multi-Objective Algorithm
    
    properties(GetAccess = 'public', SetAccess = 'private')
        fitness;   % fitness function
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = SMSEMOA_Solver(elitism, mutation, fitness, crossover, mutate, max_size)
            obj = obj@Genetic_Solver(elitism, mutation, crossover, mutate, max_size);
            obj.fitness = fitness;
        end
        
        function values = getFitness ( obj, J )
        % In SMS-EMOA, a solution is ranked higher if removing it from the 
        % population the fitness decreases.
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
