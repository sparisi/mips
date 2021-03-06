classdef CREPS_Solver < handle
% Contextual Relative Entropy Policy Search.
% The value function is linear in the parameters: V(s) = theta' * Phi(s).
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
        basis   % Basis function to approximate the value function
        eta     % Lagrangian (KL)
        theta   % Lagrangian (features)
        tolKL = 0.1; % Tolerance of the KL error 
        tolSF = 1e-5; % Tolerance of feature matching error
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = CREPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1e3;
            obj.theta = rand(bfs(),1)-0.5;
        end
        
        %% CORE
        function [sampling, divKL] = step(obj, J, Actions, Phi, sampling, W)
            if nargin < 6, W = ones(1, size(J,2)); end % IS weights
            [d, divKL] = obj.optimize(J, Phi, W);
            sampling = sampling.weightedMLUpdate(d, Actions, [ones(1,size(Phi,2)); Phi]);
        end
        
        function [d, divKL] = optimize(obj, J, Phi, W)
            if nargin < 4, W = ones(1, size(J,2)); end % IS weights
            
            % Optimization problem settings
            options_eta = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 100, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);
            
            options_theta = optimoptions('fminunc', ...
                'Algorithm', 'trust-region', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 100, ...
                'TolX', 10^-8, 'TolFun', 10^-12, 'MaxIter', 100);

            % Hyperparams
            maxIter = 100;
            numStepsNoKL = 0;

            validKL = false;
            validSF = false;
            
            % Iteratively solve fmincon for eta and theta separately
            for i = 1 : maxIter
                
                V = obj.theta' * Phi; % V(c)

                if ~validKL || numStepsNoKL > 5 % If we skipped the KL optimization more than 5 times, redo it
                    % Here theta is constant, so we can pass directly V
                    obj.eta = fmincon(@(eta)obj.dual_eta(eta,J,V,W), obj.eta, ...
                        [], [], [], [], 1e-8, 1e8, [], options_eta);
                    
                    % Compute the weights
                    advantage = J - V;
                    d = W .* exp( (advantage - max(advantage)) / obj.eta );
                    
                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = abs(divKL - obj.epsilon);
                    validKL = error < obj.tolKL * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, Phi, pWeighting),2) - mean(Phi,2);
                    validSF = max(abs(featureDiff)) < obj.tolSF;
                    numStepsNoKL = 0;
                else
                    % KL is still valid, skip KL optimization
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF
                    % Here theta is not constant
                    obj.theta = fminunc(@(theta)obj.dual_theta(theta,obj.eta,J,Phi,W), ...
                        obj.theta, options_theta);
                    
                    % Compute the weights
                    advantage = J - obj.theta' * Phi;
                    d = W .* exp( (advantage - max(advantage)) / obj.eta );
                    
                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = abs(divKL - obj.epsilon);
                    validKL = error < obj.tolKL * obj.epsilon;
                    featureDiff = sum(bsxfun(@times, Phi, pWeighting),2) - mean(Phi,2);
                    validSF = max(abs(featureDiff)) < obj.tolSF;
                end
                
                if validSF && validKL
                    break
                end
                
                if i == maxIter
                    warning('REPS could not satisfy the constraints within max iterations.')
                end
            end
        end
        
        %% DUAL FUNCTIONS
        function [g, gd] = dual_full(obj, params, J, Phi, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights
            
            theta = params(1:end-1);
            eta = params(end);
            
            V = theta' * Phi;
            n = sum(W);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = W .* exp( ( advantage - maxAdvantage ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsA = sum( weights .* (advantage - maxAdvantage) );
            meanFeatures = mean(Phi,2);
            sumWeightsPhi = ( weights * Phi' )';
            
            % Dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % Gradient wrt theta and eta
            gd = [meanFeatures - sumWeightsPhi / sumWeights;
                obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights)];
        end
        
        function [g, gd, h] = dual_eta(obj, eta, J, V, W)
            if nargin < 5, W = ones(1, size(J,2)); end % IS weights

            n = sum(W);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = W .* exp( ( advantage - maxAdvantage ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsA = sum( weights .* (advantage - maxAdvantage) );
            sumWeightsAA = sum( weights .* (advantage - maxAdvantage).^2 );
            
            % Dual function
            g = eta * obj.epsilon + mean(V,2) + eta * log(sumWeights/n) + maxAdvantage;
            % Gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights);
            % Hessian
            h = ( sumWeightsAA * sumWeights - sumWeightsA^2 ) / ( eta^3 * sumWeights^2 );
        end
        
        function [g, gd, h] = dual_theta(obj, theta, eta, J, Phi, W)
            if nargin < 6, W = ones(1, size(J,2)); end % IS weights

            V = theta' * Phi;
            n = sum(W);
            advantage = J - V;
            maxAdvantage = max(advantage);
            weights = W .* exp( ( advantage - maxAdvantage ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            meanFeatures = mean(Phi,2);
            sumPhiWeights = (Phi * weights');
            sumPhiWeightsPhi = Phi * bsxfun( @times, weights', Phi' );
            
            % Dual function
            g = eta * obj.epsilon + theta' * meanFeatures + eta * log(sumWeights/n) + maxAdvantage;
            % Gradient wrt theta
            gd = meanFeatures - sumPhiWeights / sumWeights;
            % Hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta;
        end

        %% GET V-FUNCTION
        function V = getV(obj, ctx)
            V = obj.theta'*obj.basis(ctx);
        end

        %% PLOTTING
        function plotV(obj, ctxLB, ctxUB, type)
            if length(ctxLB) > 2, return, end
            if sum(isinf(ctxLB) | isinf(ctxUB)) > 0, return, end
            if nargin < 4, type = 'contourf'; end

            if length(ctxLB) == 1
                n = 100;
                s = linspace(ctxLB(1),ctxUB(1),n);
                V = obj.getV(s);

                fig = findobj('type','figure','name','V-function');
                if isempty(fig)
                    figure('Name','V-function')
                    title('V-function')
                    plot(s,V)
                else
                    fig.CurrentAxes.Children.YData = V;
                    drawnow limitrate
                end

            elseif length(ctxLB) == 2
                n = 30;
                x = linspace(ctxLB(1),ctxUB(1),n);
                y = linspace(ctxLB(2),ctxUB(2),n);
                [X, Y] = meshgrid(x,y);
                s = [X(:), Y(:)]';
                V = obj.getV(s);
                if strcmp(type, 'contourf')
                    updatecontourf('V-function', X, Y, reshape(V,n,n))
                elseif strcmp(type, 'surf')
                    updatesurf('V-function', X, Y, reshape(V,n,n))
                else
                    error('Unknown plot type.')
                end
            end
        end

    end
    
end
