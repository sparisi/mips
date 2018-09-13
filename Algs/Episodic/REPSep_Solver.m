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
            obj.eta = 1e3;
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
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 500, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

            obj.eta = fmincon(@(eta)obj.dual(eta,J,W), ...
                obj.eta, [], [], [], [], 1e-8, 1e8, [], options);
            
            % Compute weights for weighted ML update
            d = W .* exp( (J - max(J)) / obj.eta );

            % Compute KL divergence
            qWeighting = W;
            pWeighting = d;
            divKL = kl_mle(pWeighting, qWeighting);
        end
        
        %% DUAL FUNCTION
        function [g, gd, h] = dual(obj, eta, J, W)
            if nargin < 4, W = ones(1,size(J,2)); end % IS weights
            n = sum(W);
            
            maxR = max(J);
            weights = W .* exp( ( J - maxR ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsR = sum( weights .* (J - maxR) );
            sumWeightsRR = sum( weights .* (J - maxR).^2 );

            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxR; % dual
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsR / (eta * sumWeights); % gradient
            h = ( sumWeightsRR * sumWeights - sumWeightsR^2 ) / ( eta^3 * sumWeights^2 ); % hessian
        end

    end
    
end
