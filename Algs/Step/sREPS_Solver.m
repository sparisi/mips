classdef sREPS_Solver < handle
% Step-based Relative Entropy Policy Search.
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
        basis   % Basis function to approximate the value function
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = sREPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, J, Actions, Phi, PhiN, policy, W)
            if nargin < 7, W = ones(1, size(J,2)); end % IS weights
            [d, divKL] = obj.optimize(J, Phi, PhiN, W);
            policy = policy.weightedMLUpdate(d, Actions, Phi);
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, J, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            
            [dphi, N] = size(Phi);
            
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

            lowerBound_theta = -ones(dphi, 1) * 1e8;
            upperBound_theta = ones(dphi, 1) * 1e8;
            lowerBound_eta = 1e-8;
            upperBound_eta = 1e8;
            theta = ones(dphi,1);
            eta = 1;
            
            maxIter = 100;
            validKL = false;
            validSF = false;
            numStepsNoKL = 0;
            
            % Iteratively solve fmincon for eta and theta separately
            for i = 1 : maxIter
                if ~validKL || numStepsNoKL < 5
                    eta = fmincon(@(eta)obj.dual_eta(eta,theta,J,Phi,PhiN,W), ...
                        eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);
                    
                    % Numerical trick
                    berror = J - theta' * Phi + theta' * mean(PhiN,2);
                    maxBerror = max(berror);
                    
                    % Compute the weights
                    d = W .* exp( (berror - maxBerror) / eta );
                    
                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < 0.1 * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, Phi, pWeighting),2) - mean(Phi,2);
                    validSF = max(abs(featureDiff)) < 0.1;
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF
                    theta = fmincon(@(theta)obj.dual_theta(theta,eta,J,Phi,PhiN,W), ...
                        theta, [], [], [], [], lowerBound_theta, upperBound_theta, [], options);
                    
                    % Numerical trick
                    berror = J - theta' * Phi + theta' * mean(PhiN,2);
                    maxBerror = max(berror);
                    
                    % Compute the weights
                    d = W .* exp( (berror - maxBerror) / eta );
                    
                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < 0.1 * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, Phi, pWeighting),2) - mean(Phi,2);
                    validSF = max(abs(featureDiff)) < 0.1;
                end
                
                if validSF && validKL
                    break
                end
            end
        end
        
        %% DUAL FUNCTIONS
        function [g, gd, h] = dual_eta(obj, eta, theta, J, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(J,2)); end % IS weights

            V = theta' * Phi;
            VNavg = theta' * mean(PhiN,2);
            n = sum(W);
            berror = J - V + VNavg;
            maxBerror = max(berror);
            weights = W .* exp( ( berror - maxBerror ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (berror - maxBerror) );
            sumWeightsVSquare = sum( weights .* (berror - maxBerror).^2 );
            
            % dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxBerror;
            % gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights);
            % hessian
            h = ( sumWeightsVSquare * sumWeights - sumWeightsV^2 ) / ( eta^3 * sumWeights^2 );
        end
        
        function [g, gd, h] = dual_theta(obj, theta, eta, J, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(J,2)); end % IS weights

            V = theta' * Phi;
            VNavg = theta' * mean(PhiN,2);
            n = sum(W);
            berror = J - V + VNavg;
            maxBerror = max(berror);
            weights = W .* exp( ( berror - maxBerror ) / eta ); % numerical trick
            sumWeights = sum(weights);
            PhiDiff = bsxfun(@plus, mean(PhiN,2), - Phi);
            sumPhiWeights = (PhiDiff * weights');
            sumPhiWeightsPhi = PhiDiff * bsxfun( @times, weights', PhiDiff' );
            
            % dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxBerror;
            % gradient wrt theta
            gd = sumPhiWeights / sumWeights;
            % hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta;
        end

    end
    
end
