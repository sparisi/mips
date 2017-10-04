classdef LQR_v3 < MDP
% As LQR_V2, but with multiple goal states. The agent must go to the
% closest goal state.
    
    %% Properties
    properties
        % Environment variables
        A
        B
        x0
        Q
        R
        goals
        
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
        function obj = LQR_v3(dim, goals)
            obj.A = eye(dim);
            obj.B = eye(dim);
            obj.x0 = 10*ones(dim,1);
            obj.Q = eye(dim);
            obj.R = eye(dim);
            obj.goals = goals;
            
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
            for i = 1 : size(obj.goals,2)
                dist = bsxfun(@minus,state,obj.goals(:,i));
                rewardS(i,:) = -sum(bsxfun(@times, dist'*obj.Q, dist'), 2)';
            end
            
            reward = min(rewardS,[],1) ...
                -sum(bsxfun(@times, action'*obj.R, action'), 2)';
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            if obj.dstate > 2, return, end
            
            obj.handleEnv = figure(); hold all

            if obj.dstate == 2
                plot(obj.goals(1,:), obj.goals(2,:),...
                    'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
                obj.handleAgent = plot(-1,-1,...
                    'ro','MarkerSize',8,'MarkerFaceColor','r');
                axis([-20 20 -20 20])
            else
                plot([-20 20],[1 1],...
                    '-b','MarkerSize',10,'MarkerEdgeColor','b','LineWidth',2,'MarkerFaceColor','b')
                plot(obj.goals,ones(length(obj.goals)),...
                    'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
                obj.handleAgent = plot(-1,1,...
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