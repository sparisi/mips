classdef LQR_v2 < MDP
% As LQR, but single-objective. It is the classic LQR.
    
    %% Properties
    properties
        % Environment variables
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
    end
    
    methods

        %% Constructor
        function obj = LQR_v2(dim)
            obj.A = eye(dim);
            obj.B = eye(dim);
            obj.x0 = 10*ones(dim,1);
            obj.Q = eye(dim);
            obj.R = eye(dim);
            
            obj.dstate = dim;
            obj.daction = dim;
            obj.dreward = 1;
            
            % Bounds
            obj.stateLB = -inf(dim,1);
            obj.stateUB = inf(dim,1);
            obj.actionLB = -inf(dim,1);
            obj.actionUB = inf(dim,1);
            obj.rewardLB = -inf;
            obj.rewardUB = 0;
        end
        
        %% Simulator
        function state = init(obj, n)
%             state = repmat(obj.x0,1,n); % Fixed
            state = myunifrnd(-obj.x0,obj.x0,n); % Random
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstate = size(state,2);
            absorb = false(1,nstate);
            nextstate = obj.A*state + obj.B*action;
            reward = -sum(bsxfun(@times, state'*obj.Q, state'), 2)' ...
                -sum(bsxfun(@times, action'*obj.R, action'), 2)';
        end
        
        %% Closed form
        function P = riccati(obj, K)
            g = obj.gamma;
            A = obj.A;
            B = obj.B;
            R = obj.R;
            Q = obj.Q;
            I = eye(obj.dstate);

            if isequal(A, B, I)
                P = (Q + K * R * K) / (I - g * (I + 2 * K + K^2));
            else
                tolerance = 0.0001;
                converged = false;
                P = I;
                Pnew = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*PB*K + K'*R*K;
                while ~converged
                    P = Pnew;
                    Pnew = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
                    converged = max(abs(P(:)-Pnew(:))) < tolerance;
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

            if g == 1
                J(i) = - trace(Sigma*(R+B'*P*B));
            else
                J(i) = - (x0'*P*x0 + (1/(1-g))*trace(Sigma*(R+g*B'*P*B)));
            end
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            if obj.dstate > 2, return, end
            
            obj.handleEnv = figure(); hold all

            if obj.dstate == 2
                plot(0, 0,...
                    'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
                obj.handleAgent = plot(-1,-1,...
                    'ro','MarkerSize',8,'MarkerFaceColor','r');
                axis([-20 20 -20 20])
            else
                plot([-20 20],[0 0],...
                    '-b','MarkerSize',10,'MarkerEdgeColor','b','LineWidth',2,'MarkerFaceColor','b')
                plot(0,0,...
                    'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
                obj.handleAgent = plot(-1,0,...
                    'ro','MarkerSize',8,'MarkerFaceColor','r');
                axis([-20 20 -2 2])
            end
        end
        
        function updateplot(obj, state)
            if obj.dstate > 2, return, end

            obj.handleAgent.XData = state(1,1);
            if obj.dstate == 2, obj.handleAgent.YData = state(2,1); end
            drawnow limitrate
        end

    end
    
end