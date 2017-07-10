classdef CEM_Solver < handle
% Cross Entropy Method.

    properties
        elites % Number of samples to fit
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = CEM_Solver(elites)
            obj.elites = elites;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function policy = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            [~, idx] = sort(W.*J);
            idx = idx(end-obj.elites+1:end);
            policy = policy.weightedMLUpdate(ones(1,obj.elites), Actions(:,idx));
        end

    end
    
end
