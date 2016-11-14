classdef LQR < MOMDP
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)
    
    %% Properties
    properties
        % Environment variables
        e = 0.1;
        A
        B
        x0
        Q
        R
        
        % MDP variables
        dstate
        daction
        dreward
        isAveraged = 0;
        gamma = 0.9;
        
        % Upper/Lower Bounds
        stateLB
        stateUB
        actionLB
        actionUB
        rewardLB
        rewardUB
        
        % Multiobjective
        utopia
        antiutopia
    end
    
    methods

        %% Constructor
        function obj = LQR(dim)
            obj.A = eye(dim);
            obj.B = eye(dim);
            obj.x0 = 10*ones(dim,1);
            obj.Q = repmat({obj.e*eye(dim)},dim,1);
            obj.R = repmat({(1-obj.e)*eye(dim)},dim,1);
            for i = 1 : dim
                obj.Q{i}(i,i) = 1-obj.e;
                obj.R{i}(i,i) = obj.e;
            end
            
            obj.dstate = dim;
            obj.daction = dim;
            obj.dreward = dim;
            
            % Bounds
            obj.stateLB = -inf(dim,1);
            obj.stateUB = inf(dim,1);
            obj.actionLB = -inf(dim,1);
            obj.actionUB = inf(dim,1);
            obj.rewardLB = -inf(dim,1);
            obj.rewardUB = zeros(dim,1);
            
            % Multiobjective
            switch dim
                case 2
                    obj.utopia = -150*ones(1,dim);
                    obj.antiutopia = -310*ones(1,dim);
                case 3
                    obj.utopia = -195*ones(1,dim);
                    obj.antiutopia = -360*ones(1,dim);
                case 5
                    obj.utopia = -283*ones(1,dim);
                    obj.antiutopia = -436*ones(1,dim);
                otherwise
                    warning('Multiobjective framework not available for the desired number of objective.')
            end
        end
        
        %% Simulator
        function state = init(obj, n)
            state = repmat(obj.x0,1,n);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstate = size(state,2);
            absorb = false(1,nstate);
            nextstate = obj.A*state + obj.B*action;
            reward = zeros(obj.dreward,nstate);
            
            for i = 1 : obj.dreward
                reward(i,:) = -sum(bsxfun(@times, state'*obj.Q{i}, state'), 2)' ...
                    -sum(bsxfun(@times, action'*obj.R{i}, action'), 2)';
            end
        end
        
        %% Multiobjective
        function [front, weights] = truefront(obj)
            try
                front = dlmread(['lqr_front' num2str(obj.dreward) 'd.dat']);
                weights = dlmread(['lqr_w' num2str(obj.dreward) 'd.dat']);
                warning('This frontier was obtained with a Gaussian policy with fixed identity covariance!')
            catch
                warning('Multiobjective framework not available for this number of objective.')
            end
        end

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(front, varargin{:});
            xlabel 'Obj 1'
            ylabel 'Obj 2'
            zlabel 'Obj 3'
        end
        
        %% Closed form
        function P = riccati(obj, K)
            g = obj.gamma;
            A = obj.A;
            B = obj.B;
            R = obj.R;
            Q = obj.Q;
            I = eye(obj.dreward);

            P = zeros(obj.dreward,obj.dreward,obj.dreward);
            
            for i = 1 : obj.dreward
                if isequal(A, B, I)
                    P(:,:,i) = (Q{i} + K * R{i} * K) / (I - g * (I + 2 * K + K^2));
                else
                    tolerance = 0.0001;
                    converged = false;
                    P(:,:,i) = I;
                    Pnew(:,:,i) = Q{i} + g*A'*P(:,:,i)*A + g*K'*B'*P(:,:,i)*A + g*A'*P(:,:,i)*B*K + g*K'*B'*P(:,:,i)*B*K + K'*R{i}*K;
                    while ~converged
                        P(:,:,i) = Pnew(:,:,i);
                        Pnew(:,:,i) = Q{i} + g*A'*P(:,:,i)*A + g*K'*B'*P(:,:,i)*A + g*A'*P(:,:,i)*B*K + g*K'*B'*P(:,:,i)*B*K + K'*R{i}*K;
                        converged = max(abs(P(:)-Pnew(:))) < tolerance;
                    end
                end
            end
        end
        
        function J = avg_return(obj, K, Sigma)
            P = obj.riccati(K);
            J = zeros(obj.dreward,1);
            B = obj.B;
            R = obj.R;
            g = obj.gamma;
            x0 = obj.x0;

            for i = 1 : obj.dreward
                if g == 1
                    J(i) = - trace(Sigma*(R{i}+B'*P(:,:,i)*B));
                else
                    J(i) = - (x0'*P(:,:,i)*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*B'*P(:,:,i)*B)));
                end
            end
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            warning('This MDP does not support plotting!')
        end
        
        function updateplot(obj, state)
            warning('This MDP does not support plotting!')
        end

    end
    
end