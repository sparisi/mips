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
        eta     % Lagrangian
        theta   % Lagrangian
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = sREPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1;
            obj.theta = ones(bfs()+1,1);
        end
        
        %% PERFORM AN OPTIMIZATION STEP
        function [policy, divKL] = step(obj, Q, Actions, Phi, PhiN, policy, W)
            if nargin < 7, W = ones(1, size(Q,2)); end % IS weights
            [d, divKL] = obj.optimize(Q, Phi, PhiN, W);
            policy = policy.weightedMLUpdate(d, Actions, Phi);
        end
        
        %% CORE
        function [d, divKL] = optimize(obj, Q, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(Q,2)); end % IS weights
            
            % Optimization problem settings
            options = optimset('Algorithm', 'interior-point', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 10 * 5, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 50);
            
%             options = optimset('Algorithm', 'trust-region-reflective', ...
%                 'GradObj', 'on', ...
%                 'Hessian', 'user-supplied', ...
%                 'Display', 'off', ...
%                 'MaxFunEvals', 10 * 5, ...
%                 'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 50);

            lowerBound_theta = -ones(size(Phi,1), 1) * 1e8;
            upperBound_theta = ones(size(Phi,1), 1) * 1e8;
            lowerBound_eta = 1e-8;
            upperBound_eta = 1e8;
            
            maxIter = 100;
            validKL = false;
            validSF = false;
            numStepsNoKL = 0;
            
            % Iteratively solve fmincon for eta and theta separately
            for i = 1 : maxIter
                if ~validKL || numStepsNoKL < 5
                    obj.eta = fmincon(@(eta)obj.dual_eta(eta,obj.theta,Q,Phi,PhiN,W), ...
                        obj.eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);
                    
                    % Numerical trick
                    berror = Q - obj.theta' * Phi + obj.theta' * mean(PhiN,2);
                    maxBerror = max(berror);
                    
                    % Compute the weights
                    d = W .* exp( (berror - maxBerror) / obj.eta );
                    
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
                    obj.theta = fmincon(@(theta)obj.dual_theta(theta,obj.eta,Q,Phi,PhiN,W), ...
                        obj.theta, [], [], [], [], lowerBound_theta, upperBound_theta, [], options);
                    
                    % Numerical trick% 
                    berror = Q - obj.theta' * Phi + obj.theta' * mean(PhiN,2);

                    maxBerror = max(berror);
                    
                    % Compute the weights
                    d = W .* exp( (berror - maxBerror) / obj.eta );
                    
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
        function [g, gd, h] = dual_eta(obj, eta, theta, Q, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(Q,2)); end % IS weights

            V = theta' * Phi;
            VNavg = theta' * mean(PhiN,2);            

            n = sum(W);
            berror = Q - V + VNavg;
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
        
        function [g, gd, h] = dual_theta(obj, theta, eta, Q, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(Q,2)); end % IS weights

            V = theta' * Phi;
            VNavg = theta' * mean(PhiN,2);
            n = sum(W);
            berror = Q - V + VNavg;
            maxBerror = max(berror);
            weights = W .* exp( ( berror - maxBerror ) / eta ); % numerical trick
            sumWeights = sum(weights);
            bErrDeriv = bsxfun(@plus, mean(PhiN,2), - Phi);
            sumPhiWeights = (bErrDeriv * weights');
            sumPhiWeightsPhi = bErrDeriv * bsxfun( @times, weights', bErrDeriv' );
            
            % dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxBerror;
            % gradient wrt theta
            gd = sumPhiWeights / sumWeights;
            % hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta;
        end
        
        %% PLOTTING
        function plotV(obj, stateLB, stateUB)
            n = 30;
            x = linspace(stateLB(1),stateUB(1),n);
            y = linspace(stateLB(2),stateUB(2),n);
            [X, Y] = meshgrid(x,y);
            s = [X(:), Y(:)]';
            V = obj.theta'*[ones(1,size(s,2)); obj.basis(s)];
            subimagesc('V-function',x,y,V)
        end

    end
    
end
