classdef REPSavg_Solver < handle
% Step-based Relative Entropy Policy Search for average rewards.
% The value function is linear in the parameters: V(s) = theta' * Phi(s).
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
        eta     % Lagrangian (KL)
        theta   % Lagrangian (features)
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = REPSavg_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1;
            obj.theta = ones(bfs(),1);
        end

        %% CORE
        function [d, divKL] = optimize(obj, R, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(R,2)); end % IS weights

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

            % Hyperparams
            maxIter = 100;
            numStepsNoKL = 0;
            tolSF = 0.0001;
            tolKL = 0.1;

            validKL = false;
            validSF = false;

            % Iteratively solve fmincon for eta and theta separately
            for i = 1 : maxIter

                V = obj.theta' * Phi; % V(s)
                VN = obj.theta' * PhiN; % V(s')
                
                if ~validKL || numStepsNoKL > 5 % If we skipped the KL optimization more than 5 times, redo it
                    % Here theta is constant, so we can pass directly V and VN
                    obj.eta = fmincon(@(eta)obj.dual_eta(eta,R,V,VN,W), ...
                        obj.eta, [], [], [], [], lowerBound_eta, upperBound_eta, [], options);

                    % Compute the weights
                    bError = R + VN - V;
                    d = W .* exp( (bError - max(bError)) / obj.eta );

                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < tolKL * obj.epsilon;
                    featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
                    validSF = max(abs(featureDiff)) < tolSF;
                    numStepsNoKL = 0;
                else
                    % KL is still valid, skip KL optimization
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF
                    % Here theta is not constant
                    obj.theta = fmincon(@(theta)obj.dual_theta(theta,obj.eta,R,Phi,PhiN,W), ...
                        obj.theta, [], [], [], [], lowerBound_theta, upperBound_theta, [], options);

                    % Compute the weights
                    bError = R + obj.theta' * (PhiN - Phi);
                    d = W .* exp( (bError - max(bError)) / obj.eta );

                    % Check conditions
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    error = divKL - obj.epsilon;
                    validKL = error < tolKL * obj.epsilon;
                    featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
                    validSF = max(abs(featureDiff)) < tolSF;
                end
                
                if validSF && validKL
                    break
                end
                
            end
            
            if i == maxIter
                warning('REPS could not satisfy the constraints within max iterations.')
            end
            
        end

        %% DUAL FUNCTIONS
        function [g, gd, h] = dual_eta(obj, eta, R, V, VN, W)
            if nargin < 6, W = ones(1, size(R,2)); end % IS weights
            
            n = sum(W);
            bError = R + VN - V;
            maxBError = max(bError);
            weights = W .* exp( ( bError - maxBError ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsB = (bError - maxBError) * weights';
            sumWeightsBB = (bError - maxBError).^2 * weights';

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxBError;
            % Gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsB / (eta * sumWeights);
            % Hessian
            h = (sumWeightsBB * sumWeights - sumWeightsB^2) / (eta^3 * sumWeights^2);
        end

        function [g, gd, h] = dual_theta(obj, theta, eta, R, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(R,2)); end % IS weights

            n = sum(W);
            PhiDiff = PhiN - Phi;
            bError = R + theta' * PhiDiff;
            maxBError = max(bError);
            weights = W .* exp( ( bError - maxBError ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumPhiWeights = PhiDiff * weights';
            sumPhiWeightsPhi = PhiDiff * bsxfun( @times, weights', PhiDiff' );

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxBError;
            % Gradient wrt theta
            gd = sumPhiWeights / sumWeights;
            % Hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta;
        end
        
        %% GET V-FUNCTION
        function V = getV(obj, state)
            V = obj.theta'*obj.basis(state);
        end

        %% PLOTTING
        function plotV(obj, stateLB, stateUB)
            if length(stateLB) > 2, return, end

            fig = findobj('type','figure','name','V-function');
            if isempty(fig)
                figure('Name','V-function')
                if length(stateLB) == 2
                    axis([stateLB(1) stateUB(1) stateLB(2) stateUB(2)])
                end
            else
                figure(fig)
            end
            
            if length(stateLB) == 1
                n = 100;
                s = linspace(stateLB(1),stateUB(1),n);
                V = obj.getV(s);
                plot(s,V)
            elseif length(stateLB) == 2
                n = 30;
                x = linspace(stateLB(1),stateUB(1),n);
                y = linspace(stateLB(2),stateUB(2),n);
                [X, Y] = meshgrid(x,y);
                s = [X(:), Y(:)]';
                V = obj.getV(s);
%                 imagesc('XData',x,'YData',y,'CData',reshape(V,n,n))
                surf(X,Y,reshape(V,n,n))
                colorbar
            end
            
            xlabel x
            ylabel y
            
            drawnow limitrate
        end

    end

end
