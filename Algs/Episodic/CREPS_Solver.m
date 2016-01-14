classdef CREPS_Solver < handle
% Contextual Relative Entropy Policy Search.
%
% =========================================================================
% REFERENCE
% M P Deisenroth, G Neumann, J Peters
% A Survey on Policy Search for Robotics, Foundations and Trends in
% Robotics (2013)

    properties
        epsilon % KL divergence bound
        basis   % Basis function to approximate the value function
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = CREPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, J, Phi)
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
                    eta = fmincon(@(eta)obj.dual_eta(eta,theta,J,Phi), ...
                        eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);
                    
                    % Numerical trick
                    advantage = J - theta' * Phi;
                    maxAdvantage = max(advantage);
                    
                    % Compute the weights
                    d = exp( (advantage - maxAdvantage) / eta );
                    
                    % Check conditions
                    qWeighting = ones(1,N);
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = getKL(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < 0.1 * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, Phi, pWeighting),2) - mean(Phi,2);
                    validSF = max(abs(featureDiff)) < 0.1;
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF
                    theta = fmincon(@(theta)obj.dual_theta(theta,eta,J,Phi), ...
                        theta, [], [], [], [], lowerBound_theta, upperBound_theta, [], options);
                    
                    % Numerical trick
                    advantage = J - theta' * Phi;
                    maxAdvantage = max(advantage);
                    
                    % Compute the weights
                    d = exp( (advantage - maxAdvantage) / eta );
                    
                    % Check conditions
                    qWeighting = ones(1,N);
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = getKL(pWeighting, qWeighting);
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
        function [g, gd] = dual_full(obj, params, J, Phi)
            theta = params(1:end-1);
            eta = params(end);
            
            V = theta' * Phi;
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            meanFeatures = mean(Phi,2);
            sumWeightsPhi = ( weights * Phi' )';
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt theta and eta
            gd = [meanFeatures - sumWeightsPhi / sumWeights;
                obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights)];
        end
        
        function [g, gd, h] = dual_eta(obj, eta, theta, J, Phi)
            V = theta' * Phi;
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            sumWeightsV = sum( weights .* (advantage - maxAdvantage) );
            sumWeightsVSquare = sum( weights .* (advantage - maxAdvantage).^2 );
            meanFeatures = mean(Phi,2);
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsV / (eta * sumWeights);
            % hessian
            h = ( sumWeightsVSquare * sumWeights - sumWeightsV^2 ) / ( eta^3 * sumWeightsV^2 );
        end
        
        function [g, gd, h] = dual_theta(obj, theta, eta, J, Phi)
            V = theta' * Phi;
            n = length(J);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = exp( ( advantage - maxAdvantage ) / eta ); % numerical trick
            sumWeights = sum(weights);
            meanFeatures = mean(Phi,2);
            sumWeightsPhi = ( weights * Phi' )';
            sumPhiWeights = (Phi * weights');
            sumPhiWeightsPhi = Phi * bsxfun( @times, weights', Phi' );
            
            % dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % gradient wrt theta
            gd = meanFeatures - sumWeightsPhi / sumWeights;
            % hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumWeightsPhi') / sumWeights^2 / eta;
        end

    end
    
end