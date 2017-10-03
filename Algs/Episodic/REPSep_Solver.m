classdef REPSep_Solver < handle
% Relative Entropy Policy Search for bandits (episodic problems).
% It supports Importance Sampling (IS).
%
% =========================================================================
% REFERENCE
% M P Deisenroth, G Neumann, J Peters
% A Survey on Policy Search for Robotics, Foundations and Trends in
% Robotics (2013)
%
% C Daniel, G Neumann, J Peters
% Learning Concurrent Motor Skills in Versatile Solution Spaces (2012)

    properties
        epsilon % KL divergence bound
        eta     % Lagrangian
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPSep_Solver(epsilon)
            obj.epsilon = epsilon;
            obj.eta = 1;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, policy, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            [d, divKL] = obj.optimize(J, W);
            policy = policy.weightedMLUpdate(d, Actions);
        end

        %% CORE
        function [d, divKL] = optimize(obj, J, W)
            if nargin < 3, W = ones(1,size(J,2)); end % IS weights
            
            % Optimization problem settings
            options = optimset('GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 300 * 5, ...
                'Algorithm', 'interior-point', ...
                'TolX', 10^-8, ...
                'TolFun', 10^-12, ...
                'MaxIter', 300);
            lowerBound = 1e-8; % eta > 0
            upperBound = 1e8; % eta < inf
            eta0 = 1;
            obj.eta = fmincon(@(eta)obj.dual(eta,J,W), ...
                eta0, [], [], [], [], lowerBound, upperBound, [], options);
            
            % Compute weights for weighted ML update
            d = W .* exp( (J - max(J)) / obj.eta );

            % Compute KL divergence
            qWeighting = W;
            pWeighting = d;
            divKL = kl_mle(pWeighting, qWeighting);
        end
        
        %% DUAL FUNCTION
        function [g, gd] = dual(obj, eta, J, W)
            if nargin < 4, W = ones(1,size(J,2)); end % IS weights
            
            % Numerical trick
            maxJ = max(J);
            J = J - maxJ;
            
            A = sum(W .* exp(J / eta)) / sum(W);
            B = sum(W .* exp(J / eta) .* J) / sum(W);
            
            g = eta * obj.epsilon + eta * log(A) + maxJ; % dual function
            gd = obj.epsilon + log(A) - B / (eta * A);   % gradient
        end

    end
    
end
