classdef REPS_Solver2 < handle
% Like REPS_Solver, but eta and theta are optimized together.

    properties
        epsilon       % KL divergence bound
        basis         % Basis function to approximate the value function
        eta           % Lagrangian (KL)
        theta         % Lagrangian (features)
        l2_reg        % Regularizer for theta
        tolKL = 0.1;  % Tolerance of the KL error 
        tolSF = 1e-5; % Tolerance of feature matching error
        verbose = 1;  % 1 to display inner loop statistics
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = REPS_Solver2(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1e3;
            obj.theta = rand(bfs(),1)-0.5;
            obj.l2_reg = 0;
        end

        %% CORE
        function [d, divKL, featureDiff] = optimize(obj, R, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(R,2)); end % IS weights

            % Optimization problem settings
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 300, ...
                'TolX', 10^-12, 'TolFun', 10^-12, 'MaxIter', 300);
            
            maxIter = 10;

            for iter = 1 : maxIter
                if obj.verbose, fprintf('     %d | ', iter); end

                vars = fmincon(@(vars)obj.dual(vars,R,Phi,PhiN,W), [obj.eta; obj.theta], ...
                            [], [], [], [], 1e-8, inf, [], options);
                obj.eta = vars(1);
                obj.theta = vars(2:end);

                % Compute the weights
                A = R + obj.theta' * (PhiN - Phi);
                d = W .* exp( (A - max(A)) / obj.eta );

                % Check conditions
                qWeighting = W;
                pWeighting = d;
                pWeighting = pWeighting / sum(pWeighting);
                divKL = kl_mle(pWeighting, qWeighting);

                % Print info
                featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
                errorSF = max(abs(featureDiff));
                if obj.verbose
                    fprintf('KL: %.4f / %.4f,  FE: %e / %e,  ETA: %e\n', ...
                        divKL, (1+obj.tolKL)*obj.epsilon, errorSF, obj.tolSF, obj.eta)
                end
                
                errorKL = abs(divKL - obj.epsilon);
                validKL = errorKL < obj.tolKL * obj.epsilon;
                validSF = errorSF < obj.tolSF;
                if validKL && validSF, break, end
            end
            
            if obj.verbose, fprintf('\n'), end
            
            if iter == maxIter
                warning('REPS could not satisfy the constraints within max iterations.')
            end
        end

        %% DUAL FUNCTION
        function [g, gd] = dual(obj, vars, R, Phi, PhiN, W)
            if nargin < 6, W = ones(1, size(R,2)); end % IS weights
            eta = vars(1);
            theta = vars(2:end);

            n = sum(W);
            PhiDiff = PhiN - Phi;
            A = R + theta' * PhiDiff;
            maxA = max(A);
            weights = W .* exp( ( A - maxA ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumPhiWeights = PhiDiff * weights';
            sumWeightsA = (A - maxA) * weights';

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxA + obj.l2_reg*sum(theta.^2);
            % Gradient wrt eta and theta
            gd = [obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights)
                sumPhiWeights / sumWeights + obj.l2_reg/2*theta];
        end
        
        %% GET V-FUNCTION
        function V = getV(obj, state)
            V = obj.theta'*obj.basis(state);
        end

        %% PLOTTING
        function plotV(obj, stateLB, stateUB)
            if length(stateLB) > 2, return, end
            if sum(isinf(stateLB) | isinf(stateUB)) > 0, return, end

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
                updatecontourf('V-function', X, Y, reshape(V,n,n))
            end
        end

    end

end
