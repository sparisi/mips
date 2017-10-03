classdef REPSdisc_Solver < handle
% Step-based Relative Entropy Policy Search for discounted reward. 
% The Q-function is approximated by samples.
% It supports Importance Sampling (IS).
%
% =========================================================================
% REFERENCE
% M P Deisenroth, G Neumann, J Peters
% A Survey on Policy Search for Robotics, Foundations and Trends in
% Robotics (2013)
%
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)

    properties
        epsilon % KL divergence bound
        eta     % Lagrangian
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPSdisc_Solver(epsilon)
            obj.epsilon = epsilon;
            obj.eta = 1;
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, Q, W)
            if nargin < 3, W = ones(1, size(Q,2)); end % IS weights

            % Optimization problem settings
            options = optimset('Algorithm', 'interior-point', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 10 * 5, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 10);
            
%             options = optimset('Algorithm', 'trust-region-reflective', ...
%                 'GradObj', 'on', ...
%                 'Hessian', 'user-supplied', ...
%                 'Display', 'off', ...
%                 'MaxFunEvals', 10 * 5, ...
%                 'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 10);

            lowerBound_eta = 1e-8;
            upperBound_eta = 1e8;
            
            maxIter = 100;
            validKL = false;
            
            for i = 1 : maxIter
                if validKL, break, end

                obj.eta = fmincon(@(eta)obj.dual_eta(eta,Q,W), ...
                    obj.eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);
                
                % Compute the weights
                d = W .* exp( (Q - max(Q)) / obj.eta );
                
                % Check conditions
                qWeighting = W;
                pWeighting = d;
                pWeighting = pWeighting / sum(pWeighting);
                divKL = kl_mle(pWeighting, qWeighting);
                error = divKL - obj.epsilon;
                validKL = error < 0.1 * obj.epsilon;
            end
        end
        
        %% DUAL FUNCTIONS
        function [g, gd, h] = dual_eta(obj, eta, Q, W)
            if nargin < 4, W = ones(1, size(Q,2)); end % IS weights

            n = sum(W);
            maxQ = max(Q);
            weights = W .* exp( ( Q - maxQ ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (Q - maxQ) );
            sumWeightsVSquare = sum( weights .* (Q - maxQ).^2 );
            
            % dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxQ;
            % gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights);
            % hessian
            h = ( sumWeightsVSquare * sumWeights - sumWeightsV^2 ) / ( eta^3 * sumWeights^2 );
        end

    end
    
end