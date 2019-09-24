classdef ChainwalkCont < MDP
% Continuous chainwalk with simple dynamics.
% Starting state is always in 0.
% Goal state is also fixed.
% Reward is received only when the goal is found (which requires almost 0
% velocity).
    
    %% Properties
    properties
        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        dt = 0.05;
        tol = 0.05;
        
        % Bounds
        stateLB = [-1, -1]'; % [x, x_dot]
        stateUB = [1, 1]';
        actionLB = -1; % acc
        actionUB = 1;
        rewardLB = 0;
        rewardUB = 1;

        % Environment variables
        goal_state = 0.65;
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = zeros(2,n);
        end

        function action = parse(obj, action)
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
        end

        function nextstate = transition(obj, state, action)
            x = state(1,:);
            xd = state(2,:);
            xd_next = xd + obj.dt * action;
            x_next = x + obj.dt * xd;
            nextstate = [x_next; xd_next];
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
        function reward = reward(obj, state, action, nextstate)
            reward = zeros(1,size(nextstate,2));
            close_goal = abs(nextstate(1,:) - obj.goal_state) < obj.tol & abs(nextstate(2,:)) < obj.tol;
            reward(close_goal) = 1;
%             reward = reward - 0.001 * action.^2;
        end
        
        function absorb = isterminal(obj, state)
            absorb = abs(state(1,:) - obj.goal_state) < obj.tol & abs(state(2,:)) < obj.tol;
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            axis off
            
            plot([obj.stateLB(1),obj.stateUB(1)], [1,1], ...
                'b','LineWidth',2)
            plot(obj.goal_state,ones(length(obj.goal_state)),...
                'og','MarkerSize',14,'MarkerFaceColor','g','MarkerEdgeColor','k');
            obj.handleAgent{1} = plot(1,1,...
                'ro','MarkerSize',10,'MarkerFaceColor','r','MarkerEdgeColor','k');
        end
        
        function updateplot(obj, state)
            obj.handleAgent{1}.XData = state(1);
            drawnow limitrate
        end
        
    end
    
end