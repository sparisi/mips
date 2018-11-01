classdef Chainwalk < MDP
    
    %% Properties
    properties
        % MDP variables
        dstate = 1;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;
        
        % Bounds
        stateLB = 1;
        stateUB = 50;
        actionLB = 1;
        actionUB = 2;
        rewardLB = 0;
        rewardUB = 1;

        % Environment variables
        reward_states = [10 41];

        % Finite states and actions
        allstates = (1:50)';
        allactions = [-1 1];
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            state = randi(obj.stateUB,1,n);
        end

        function action = parse(obj, action)
            action = obj.allactions(:,action);
            noise = rand(1,size(action,2));
            action(noise < 0.1) = -action(noise < 0.1);
        end

        function nextstate = transition(obj, state, action)
            nextstate = state + action;
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
        function reward = reward(obj, state, action, nextstate)
%             reward = zeros(1,size(state,2));
%             reward(ismember(state,obj.reward_states)) = 1;

%             reward = zeros(1,size(state,2));
%             reward(min(dist,[],1)<=1) = 1;

            dist = abs(bsxfun(@minus,state,obj.reward_states'));
            reward = -min(dist,[],1);
        end
        
        function absorb = isterminal(obj, state)
            absorb = false(1,size(state,2));
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
            obj.handleAgent.XData = state(1);
            drawnow limitrate
        end
        
    end
    
end