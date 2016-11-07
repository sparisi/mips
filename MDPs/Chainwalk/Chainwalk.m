classdef Chainwalk < MDP
    
    %% Properties
    properties
        % Environment variables
        reward_states = [10 41];

        % Finite states and actions
        allstates = 1:50;
        allactions = [1 2];

        % MDP variables
        dstate = 1;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.8;
        
        % Bounds
        stateLB = 1;
        stateUB = 50;
        actionLB = 1;
        actionUB = 2;
        rewardLB = 0;
        rewardUB = 1;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj, n)
            state = randi(obj.stateUB,1,n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            n = size(state,2);
            
            steps = [-1 1]; % action mapping (left right)
            action = steps(:,action);
            noise = rand(1,n);
            action(noise < 0.1) = -action(noise < 0.1);
            nextstate = state + action;
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Reward function
            reward = zeros(1,n);
            reward(ismember(nextstate,obj.reward_states)) = 1;

            % Infinite horizon
            absorb = false(1,n);
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            axis off
            
            plot(obj.allstates,ones(1,obj.stateUB),...
                'ob','MarkerSize',10,'MarkerEdgeColor','b','LineWidth',2)
            plot(obj.reward_states,ones(length(obj.reward_states)),...
                'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2)
            obj.handleAgent = plot(1,1,...
                'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state;
            drawnow limitrate
        end
        
    end
    
end