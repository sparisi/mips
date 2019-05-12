classdef REPS_Solver < handle
% Step-based Relative Entropy Policy Search for average reward MDPs.
% This is the original REPS.
% Eta and theta are optimized separately.
% The value function is linear in the parameters: V(s) = theta' * Phi(s).
% It supports Importance Sampling (IS).
%
% =========================================================================
% REFERENCE
% J Peters, K Muelling, Y Altun
% Relative Entropy Policy Search (2010)
%
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss
% JMLR (2017)

    properties
        epsilon       % KL divergence bound
        basis         % Basis function to approximate the value function
        eta           % Lagrangian (KL)
        theta         % Lagrangian (features)
        l2_reg        % Regularizer for theta
        tolKL = 0.1;  % Tolerance of the KL error
        tolSF = 1e-6; % Tolerance of feature matching error
        verbose = 0;  % 1 to display inner loop statistics
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = REPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1e3;
            obj.theta = rand(bfs(),1)-0.5;
            obj.l2_reg = 0;
        end

        %% CORE
        function [d, divKL] = optimize(obj, R, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(R,2)); end % IS weights

            % Optimization problem settings
            options_eta = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 100, ...
                'TolX', 0, 'TolFun', 0, 'MaxIter', 10);

            options_theta = optimoptions('fminunc', ...
                'Algorithm', 'trust-region', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'StepTolerance', 0, ...
                'FunctionTolerance', 0, ...
                'MaxFunEvals', 100, ...
                'TolX', 0, 'TolFun', 0, 'MaxIter', 10);

            % Hyperparams
            maxIter = 10;
            numStepsNoKL = 0;

            validKL = false;
            validSF = false;

            % Iteratively solve fmincon for eta and theta separately
            for iter = 1 : maxIter
                if ~validKL || numStepsNoKL > 5 % If we skipped the KL optimization more than 5 times, redo it
                    if obj.verbose, fprintf('     %d | eta   - ', iter); end
                    
                    % Here theta is constant, so we can pass directly A
                    A = R + obj.theta'*(PhiN-Phi);
                    obj.eta = fmincon(@(eta)obj.dual_eta(eta,A,W), obj.eta, ...
                        [], [], [], [], 1e-8, 1e8, [], options_eta);

                    % Compute the weights
                    d = W .* exp( (A - max(A)) / obj.eta );

                    % Check constraints
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    errorKL = abs(divKL - obj.epsilon);
                    validKL = errorKL < obj.tolKL * obj.epsilon;
%                     validKL = divKL < (1+tolKL) * obj.epsilon;
                    featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
                    errorSF = max(abs(featureDiff));
                    validSF = errorSF < obj.tolSF;
                    numStepsNoKL = 0;
                    if obj.verbose
                        fprintf('KL: %.4f / %.4f,  FE: %e / %e,  ETA: %e,  MSA: %e\n', ...
                            divKL, (1+obj.tolKL)*obj.epsilon, errorSF, obj.tolSF, obj.eta, mean(A.^2))
                    end
                else
                    % KL is still valid, skip KL optimization
                    numStepsNoKL = numStepsNoKL + 1;
                end
                
                if ~validSF || iter == 1
                    if obj.verbose, fprintf('     %d | theta - ', iter); end

                    % Here theta is the variable to be learned
                    obj.theta = fminunc(@(theta)obj.dual_theta(theta,obj.eta,R,Phi,PhiN,W), ...
                        obj.theta, options_theta);

                    % Compute the weights
                    A = R + obj.theta' * (PhiN - Phi);
                    d = W .* exp( (A - max(A)) / obj.eta );

                    % Check constraints
                    qWeighting = W;
                    pWeighting = d;
                    pWeighting = pWeighting / sum(pWeighting);
                    divKL = kl_mle(pWeighting, qWeighting);
                    errorKL = abs(divKL - obj.epsilon);
                    validKL = errorKL < obj.tolKL * obj.epsilon;
%                     validKL = divKL < (1+tolKL) * obj.epsilon;
                    featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
                    errorSF = max(abs(featureDiff));
                    validSF = errorSF < obj.tolSF;
                    numStepsNoKL = 0;
                    if obj.verbose
                        fprintf('KL: %.4f / %.4f,  FE: %e / %e,  ETA: %e,  MSA: %e\n', ...
                            divKL, (1+obj.tolKL)*obj.epsilon, errorSF, obj.tolSF, obj.eta, mean(A.^2))
                    end
                end
                
                if validSF && validKL, break, end
            end
            
            if obj.verbose, fprintf('\n'), end
            
            if iter == maxIter
                warning('REPS could not satisfy the constraints within max iterations.')
            end
        end

        %% DUAL FUNCTIONS
        function [g, gd, h] = dual_eta(obj, eta, A, W)
            if nargin < 4, W = ones(1, size(A,2)); end % IS weights
            
            n = sum(W);
            maxA = max(A);
            weights = W .* exp( ( A - maxA ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumWeightsA = (A - maxA) * weights';
            sumWeightsAA = (A - maxA).^2 * weights';

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxA + obj.l2_reg*sum(obj.theta.^2);
            % Gradient wrt eta
            gd = obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights);
            % Hessian
            h = (sumWeightsAA * sumWeights - sumWeightsA^2) / (eta^3 * sumWeights^2);
        end

        function [g, gd, h] = dual_theta(obj, theta, eta, R, Phi, PhiN, W)
            if nargin < 7, W = ones(1, size(R,2)); end % IS weights

            n = sum(W);
            PhiDiff = PhiN - Phi;
            A = R + theta' * PhiDiff;
            maxA = max(A);
            weights = W .* exp( ( A - maxA ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumPhiWeights = PhiDiff * weights';
            sumPhiWeightsPhi = PhiDiff * bsxfun( @times, weights', PhiDiff' );

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxA + obj.l2_reg*sum(theta.^2);
            % Gradient wrt theta
            gd = sumPhiWeights / sumWeights + 2*obj.l2_reg*theta;
            % Hessian
            h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta + 2*obj.l2_reg;
        end
        
        %% GET V-FUNCTION
        function V = getV(obj, state)
            V = obj.theta'*obj.basis(state);
        end

        %% PLOTTING
        function plotV(obj, stateLB, stateUB, type)
            if length(stateLB) > 2, return, end
            if sum(isinf(stateLB) | isinf(stateUB)) > 0, return, end
            if nargin < 4, type = 'contourf'; end

            if length(stateLB) == 1
                n = 100;
                s = linspace(stateLB(1),stateUB(1),n);
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

            elseif length(stateLB) == 2
                n = 30;
                x = linspace(stateLB(1),stateUB(1),n);
                y = linspace(stateLB(2),stateUB(2),n);
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
