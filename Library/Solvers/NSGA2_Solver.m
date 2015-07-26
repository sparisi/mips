%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: K Deb, A Pratap, S Agarwal, T Meyarivan (2002)
% A fast and elitist multiobjective genetic algorithm: NSGA-II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef NSGA2_Solver < Genetic_Solver
    
    % Non-dominated Sorting Genetic Algorithm II
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = NSGA2_Solver(elitism, mutation, crossover, mutate, max_size)
            obj = obj@Genetic_Solver(elitism, mutation, crossover, mutate, max_size);
        end
        
        function values = getFitness ( obj, J )
        % In NSGA-II, a solution is ranked higher if it is dominated by
        % less solutions. If two solutions have the same rank, the one with
        % the higher average distance between other solutions is ranked
        % higher.
            values = nds(J);
        end
        
    end

end
