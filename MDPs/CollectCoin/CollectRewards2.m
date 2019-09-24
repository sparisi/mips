classdef CollectRewards2 < MDP
% An agent moves in a 2D environment with multiple reward states.
% If the agent walks close to a reward, it collects the reward and the 
% episode ends. The largest one is the furthest from the agent initial 
% position (which is fixed at the center of the environment).
    
    %% Properties
    properties
        reward_radius = 1;
        reward_states = [1 1; -2 3; 10 -2; 20 20]';
        reward_magnitude = [2, 4, 10, 50];

        % MDP variables
        dstate = 2;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        
        % Upper/Lower Bounds (state = position and flags for collected rewards)
        stateLB = -[20, 20]';
        stateUB = [20, 20]';
        actionLB = -[1, 1]';
        actionUB = [1, 1]';
        rewardLB = 0;
        rewardUB = 50;
    end
    
    methods

        %% Constructor
        function obj = CollectRewards2()
            obj.rewardUB = max(obj.reward_magnitude);
        end
        
        %% Simulator
        function state = init(obj, n)
            state = repmat([0;0],1,n); % Fixed
%             state = 40*(rand(2,n)-0.5);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            bounded_action = min(max(action,obj.actionLB),obj.actionUB);
            action = bounded_action;

            nstate = size(state,2);
            absorb = false(1,nstate);
            reward = zeros(1,nstate);
            nextstate = state + action;
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);

            for i = 1 : size(obj.reward_states,2)
                dist = sqrt(sum(bsxfun(@minus,state,obj.reward_states(:,i)).^2,1));
                idx_close = dist < obj.reward_radius;
                reward(idx_close) = obj.reward_magnitude(i);
                absorb(idx_close) = true; % Any reward is terminal
            end
            
            reward = reward - 0.01*sum(action.^2,1);

            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all

            obj.handleAgent{1} = plot(-1,-1,'ro','MarkerSize',8,'MarkerFaceColor','r');

            for i = 1 : size(obj.reward_states,2)
                obj.handleAgent{1+i} = plot(obj.reward_states(1,i), obj.reward_states(2,i), ...
                    'og','MarkerSize',obj.reward_magnitude(i),'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g');
            end
            axis([obj.stateLB(1) obj.stateUB(1) obj.stateLB(2) obj.stateUB(2)])
        end
        
        function updateplot(obj, state)
            obj.handleAgent{1}.XData = state(1,1);
            obj.handleAgent{1}.YData = state(2,1);
            drawnow limitrate
        end

    end
    
end