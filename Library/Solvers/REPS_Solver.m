%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: M P Deisenroth, G Neumann, J Peters (2013)
% A Survey on Policy Search for Robotics, Foundations and Trends
% in Robotics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef REPS_Solver < handle

    % Relative Entropy Policy Search

    properties(GetAccess = 'public', SetAccess = 'private')
        epsilon; % KL divergence bound
        N;       % number of samples
        policy;  % distribution for sampling the episodes
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = REPS_Solver(epsilon, N, policy)
            obj.epsilon = epsilon;
            obj.N = N;
            obj.policy = policy;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        %% CORE
        function div = step(obj, J, Actions)
            [d, div] = optimize(obj, J);
            update(obj, d, Actions);
        end
        
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
            qWeighting = ones(obj.N,1);
            pWeighting = d;
            divKL = getKL(pWeighting, qWeighting);
        end
        
        function update (obj, weights, Actions)
            obj.policy = obj.policy.weightedMLUpdate(weights, Actions);
        end

        %% DUAL FUNCTION
        function [g, gd] = dual(obj, eta, J)
            % Numerical trick
            maxJ = max(J);
            J = J - maxJ;
            
            A = sum(exp(J / eta)) / obj.N;
            B = sum(exp(J / eta) .* J) / obj.N;
            
            g = eta * obj.epsilon + eta * log(A) + maxJ; % dual function
            gd = obj.epsilon + log(A) - B / (eta * A);   % gradient
        end

    end
    
end
