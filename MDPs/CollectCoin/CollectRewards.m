classdef CollectRewards < MDP
% An agent moves in a 2D environment with multiple reward states.
% If the agent walks close to a reward, it collects the reward and the 
% reward disappears. Each reward has a different magnitude and the episode 
% ends only when the largest one is collected.
% The largest one has to be the last one in the property REWARD_MAGNITUDE.
    
    %% Properties
    properties
        reward_radius = 1;
        reward_states = [0 0; 1 1; -2 3; 10 -2; 20 20]';
        reward_magnitude = [1, 2, 4, 10, 50];

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
        function obj = CollectRewards()
            obj.dstate = obj.dstate + length(obj.reward_magnitude);
            obj.stateLB = [obj.stateLB; zeros(length(obj.reward_magnitude), 1)];
            obj.stateUB = [obj.stateUB; ones(length(obj.reward_magnitude), 1)];
            obj.rewardUB = max(obj.reward_magnitude);
        end
        
        %% Simulator
        function state = init(obj, n)
            state = myunifrnd(-[10;10],[10;10],n);
            state(3:obj.dstate,:) = 0; % No reward collected yet
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            bounded_action = min(max(action,obj.actionLB),obj.actionUB);
            action = bounded_action;

            nstate = size(state,2);
            absorb = false(1,nstate);
            reward = zeros(1,nstate);
            nextstate = state;
            nextstate(1:2,:) = state(1:2,:) + action;
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);

            for i = 1 : size(obj.reward_states,2)
                dist = sqrt(sum(bsxfun(@minus,state(1:2,:),obj.reward_states(:,i)).^2,1));
                idx_close = dist < obj.reward_radius;
                idx_notcollected = state(i+2,:) == 0;
                nextstate(i+2,idx_close & idx_notcollected) = 1;
                reward(idx_close & idx_notcollected) = obj.reward_magnitude(i);
            end
            
            reward = reward - 0.01*sum(action.^2,1);
            absorb(idx_close & idx_notcollected) = true; % Last (biggest) reward is terminal state

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
            for i = 1 : size(obj.reward_states,2)
                if state(2+i), set(obj.handleAgent{i+1},'Visible','off'); 
                else, set(obj.handleAgent{i+1},'Visible','on'); end
            end
            drawnow limitrate
        end

    end
    
end