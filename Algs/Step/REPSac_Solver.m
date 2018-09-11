classdef REPSac_Solver < handle
% Step-based Relative Entropy Policy Search with actor-critic.
% It does not constrain on the state distribution and thus it does not
% learn the value funciton, which is instead learned separately outside 
% this class.
% It maximizes E_pi[R] where R can be the Q-function, Monte-Carlo estimates
% of the return, or the advantage function, depending on which algorithm
% calls the optimizer.
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
        function obj = REPSac_Solver(epsilon)
            obj.epsilon = epsilon;
            obj.eta = 1e3;
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, R, W)
            if nargin < 3, W = ones(1, size(R,2)); end % IS weights

            % Optimization problem settings
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 500, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

            obj.eta = fmincon(@(eta)obj.dual_eta(eta,R,W), ...
                obj.eta, [], [], [], [], 1e-8, 1e8, [], options);
            
            % Compute the weights
            d = W .* exp( (R - max(R)) / obj.eta );
            
            % Check conditions
            qWeighting = W;
            pWeighting = d;
            pWeighting = pWeighting / sum(pWeighting);
            divKL = kl_mle(pWeighting, qWeighting);
        end
        
        %% DUAL FUNCTIONS
        function [g, gd, h] = dual_eta(obj, eta, R, W)
            if nargin < 4, W = ones(1, size(R,2)); end % IS weights

            n = sum(W);
            maxR = max(R);
            weights = W .* exp( ( R - maxR ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsR = sum( weights .* (R - maxR) );
            sumWeightsRR = sum( weights .* (R - maxR).^2 );
            
            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxR;
            % Gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsR / (eta * sumWeights);
            % Hessian
            h = ( sumWeightsRR * sumWeights - sumWeightsR^2 ) / ( eta^3 * sumWeights^2 );
        end

    end
    
end
