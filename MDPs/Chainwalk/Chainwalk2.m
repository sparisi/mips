classdef Chainwalk2 < MDP
% Continuous chainwalk with linear dynamics.
% Starting state is always in 0.
% Goal state is also fixed.
% A random bonus state is places either behind the start or the goal.
% The bonus is picked only if the agent is close to it with 0 velocity.
% Same for the goal position, which also ends the episode.
    
    %% Properties
    properties
        % MDP variables
        dstate = 3;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        dt = 0.1;
        
        % Bounds
        stateLB = [-1, -1, -1]'; % [x, x_dot, bonus_pos]
        stateUB = [1, 1, 1]';
        actionLB = -0.1; % acc
        actionUB = 0.1;
        rewardLB = 0;
        rewardUB = 1;

        % Environment variables
        goal_state = 0.8;
        bonus_states = [-0.2, 1];
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            state = zeros(1,n);
            r = randi(2,1,n);
            state(3,:) = obj.bonus_states(r);
        end

        function action = parse(obj, action)
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
        end

        function nextstate = transition(obj, state, action)
            x = state(1,:);
            xd = state(2,:);
            xd_next = xd + obj.dt * action;
            x_next = x + obj.dt * xd;
            close_bonus = abs(state(1,:) - state(3,:)) < 0.01 & abs(state(2,:)) < 0.01;
            bonus_next = state(3,:).*~close_bonus + obj.goal_state.*close_bonus; % if the bonus is picked, overlap it with the goal
            nextstate = [x_next; xd_next; bonus_next];
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
        function reward = reward(obj, state, action, nextstate)
            reward = zeros(1,size(state,2));
            close_bonus = abs(state(1,:) - state(3,:)) < 0.01 & abs(state(2,:)) < 0.01;
            close_goal = abs(state(1,:) - obj.goal_state) < 0.01 & abs(state(2,:)) < 0.01;
            reward(close_bonus) = 1;
            reward(close_goal) = 0.5; % this will overwrite the bonus if the bonus has been picked and placed to the goal
        end
        
        function absorb = isterminal(obj, state)
            absorb = abs(state(1,:) - obj.goal_state) < 0.01 & abs(state(2,:)) < 0.01;
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            axis off
            
            plot([obj.stateLB(1),obj.stateUB(1)], [1,1], ...
                'b','LineWidth',2)
            obj.handleAgent{2} = plot(1,1,...
                'yo','MarkerSize',14,'MarkerFaceColor','y','MarkerEdgeColor','k');
            plot(obj.goal_state,ones(length(obj.goal_state)),...
                'og','MarkerSize',14,'MarkerFaceColor','g','MarkerEdgeColor','k');
            obj.handleAgent{1} = plot(1,1,...
                'ro','MarkerSize',10,'MarkerFaceColor','r','MarkerEdgeColor','k');
        end
        
        function updateplot(obj, state)
            bonus = state(3)*(state(3)~=obj.goal_state) + obj.goal_state*(state(3)==obj.goal_state);
            obj.handleAgent{2}.XData = bonus;
            obj.handleAgent{1}.XData = state(1);
            drawnow limitrate
        end
        
    end
    
end