classdef REPS_Solver2 < handle
% Like REPS_Solver, but eta and theta are optimized together.

    properties
        epsilon       % KL divergence bound
        basis         % Basis function to approximate the value function
        eta           % Lagrangian (KL)
        theta         % Lagrangian (features)
        l2_reg        % Regularizer for theta
        verbose = 0;  % 1 to display inner loop statistics
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
        function [d, divKL] = optimize(obj, R, Phi, PhiN, W)
            if nargin < 5, W = ones(1, size(R,2)); end % IS weights

            % Optimization problem settings
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'Hessian', 'on', ...
                'MaxFunEvals', 100, ...
                'TolX', 0, 'TolFun', 0, 'MaxIter', 100);

            % Solve fmincon for eta and theta together
            params = fmincon(@(params)obj.dual(params,R,Phi,PhiN,W), [obj.eta; obj.theta], ...
                [], [], [], [], [1e-8, -inf(1,length(obj.theta))], [1e8, inf(1,length(obj.theta))], [], options);
            obj.eta = params(1);
            obj.theta = params(2:end);
            
            % Compute the weights
            A = R + obj.theta'*(PhiN-Phi);
            d = W .* exp( (A - max(A)) / obj.eta );
            
            % Check constraints
            qWeighting = W;
            pWeighting = d;
            pWeighting = pWeighting / sum(pWeighting);
            divKL = kl_mle(pWeighting, qWeighting);
            featureDiff = bsxfun(@rdivide,(Phi - PhiN)*pWeighting',std(Phi,0,2)); % Standardize
            errorSF = max(abs(featureDiff));
            if obj.verbose
                fprintf('KL: %.4f,  FE: %e,  ETA: %e,  MSA: %e\n', ...
                    divKL, errorSF, obj.eta, mean(A.^2))
                fprintf('\n')
            end
        end

        %% DUAL FUNCTIONS
        function [g, gd, h] = dual(obj, params, R, Phi, PhiN, W)
            if nargin < 6, W = ones(1, size(R,2)); end % IS weights
            eta = params(1);
            theta = params(2:end);
            
            n = sum(W);
            PhiDiff = PhiN - Phi;
            A = R + theta' * PhiDiff;
            maxA = max(A);
            weights = W .* exp( ( A - maxA ) / eta ); % Numerical trick
            sumPhiWeights = PhiDiff * weights';
            sumWeights = sum(weights);
            sumWeightsA = (A - maxA) * weights';
            
            sumPhiWeightsPhi = PhiDiff * bsxfun( @times, weights', PhiDiff' );
            sumWeightsAA = (A - maxA).^2 * weights';
            sumPhiWeightsA = PhiDiff * (weights .* (A - maxA))';
            
            h_e = (sumWeightsAA * sumWeights - sumWeightsA^2) / (eta^3 * sumWeights^2);
            h_t = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * sumPhiWeights') / sumWeights^2 / eta + 2*obj.l2_reg;
            h_et = (-sumPhiWeightsA * sumWeights + sumPhiWeights * sumWeightsA) / (eta^2 * sumWeights^2);
            
            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + maxA + obj.l2_reg*sum(obj.theta.^2);
            % Gradient wrt eta and theta
            gd = [obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights);
                sumPhiWeights / sumWeights + 2*obj.l2_reg*theta];
            % Hessian wrt eta and theta
            h = [h_e, h_et'; h_et h_t];
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
