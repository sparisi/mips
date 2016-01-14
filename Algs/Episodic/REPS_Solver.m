classdef REPS_Solver < handle
% Relative Entropy Policy Search.
%
% =========================================================================
% REFERENCE
% M P Deisenroth, G Neumann, J Peters
% A Survey on Policy Search for Robotics, Foundations and Trends in 
% Robotics (2013)

    properties
        epsilon % KL divergence bound
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPS_Solver(epsilon)
            obj.epsilon = epsilon;
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, J)
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
            eta = fmincon(@(eta)obj.dual(eta,J), ...
                eta0, [], [], [], [], lowerBound, upperBound, [], options);
            
            % Compute weights for weighted ML update
            d = exp( (J - max(J)) / eta );

            % Compute KL divergence
            qWeighting = ones(size(J));
            pWeighting = d;
            divKL = getKL(pWeighting, qWeighting);
        end
        
        %% DUAL FUNCTION
        function [g, gd] = dual(obj, eta, J)
            % Numerical trick
            maxJ = max(J);
            J = J - maxJ;
            
            N = length(J);
            A = sum(exp(J / eta)) / N;
            B = sum(exp(J / eta) .* J) / N;
            
            g = eta * obj.epsilon + eta * log(A) + maxJ; % dual function
            gd = obj.epsilon + log(A) - B / (eta * A);   % gradient
        end

    end
    
end
