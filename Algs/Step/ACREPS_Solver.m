classdef ACREPS_Solver < handle
% Actor-critic REPS. Unlike REPS, it constrains that the new state 
% distribution is the same as the old one, not that the new policy complies
% the MDP dynamics.
% The algorithm first learns Q (passed to the optimizer). Then, REPS learns
% V (seen as a baseline) and the usual eta.
% No state resets are needed.
%
% =========================================================================
% REFERENCE
% C Wirth, J Furnkranz, G Neumann
% Model-Free Preference-based Reinforcement Learning (2016)

    properties
        epsilon       % KL divergence bound
        basis         % Basis function to approximate the value function
        eta           % Lagrangian (KL)
        theta         % Lagrangian (features)
        l2_reg        % Regularizer for theta
        tolKL = 0.1;  % Tolerance of the KL error 
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = ACREPS_Solver(epsilon, bfs)
            obj.epsilon = epsilon;
            obj.basis = bfs;
            obj.eta = 1e3;
            obj.theta = rand(bfs(),1)-0.5;
            obj.l2_reg = 0;
        end

        %% CORE
        function [d, divKL] = optimize(obj, Q, Phi, W)
            if nargin < 5, W = ones(1, size(Q,2)); end % IS weights

            % Optimization problem settings
            options = optimoptions('fmincon', ...
                'Algorithm', 'trust-region-reflective', ...
                'GradObj', 'on', ...
                'Display', 'off', ...
                'MaxFunEvals', 20, ...
                'TolX', 10^-12, 'TolFun', 10^-12, 'MaxIter', 20);
            
            vars = fmincon(@(vars)obj.dual(vars,Q,Phi,W), [obj.eta; obj.theta], ...
                [], [], [], [], 1e-8, 1e8, [], options);
            obj.eta = vars(1);
            obj.theta = vars(2:end);
            
            % Compute the weights
            A = Q - obj.theta' * Phi;
            d = W .* exp( (A - max(A)) / obj.eta );
            
            % Check conditions
            qWeighting = W;
            pWeighting = d;
            pWeighting = pWeighting / sum(pWeighting);
            divKL = kl_mle(pWeighting, qWeighting);
        end

        %% DUAL FUNCTION
        function [g, gd] = dual(obj, vars, Q, Phi, W)
            if nargin < 6, W = ones(1, size(Q,2)); end % IS weights
            eta = vars(1);
            theta = vars(2:end);

            n = sum(W);
            avgPhi = mean(Phi,2);
            A = Q - theta' * Phi;
            maxA = max(A);
            weights = W .* exp( ( A - maxA ) / eta ); % Numerical trick
            sumWeights = sum(weights);
            sumPhiWeights =  - Phi * weights';
            sumWeightsA = (A - maxA) * weights';

            % Dual function
            g = eta * obj.epsilon + eta * log(sumWeights/n) + theta'*avgPhi + maxA + obj.l2_reg*sum(theta.^2);
            % Gradient wrt eta and theta
            gd = [obj.epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights)
                sumPhiWeights / sumWeights + avgPhi + obj.l2_reg*2*theta];
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
