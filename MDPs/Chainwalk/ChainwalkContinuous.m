classdef ChainwalkContinuous < MDP
    
    %% Properties
    properties
        % MDP variables
        dstate = 1;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.8;
        
        % Bounds
        stateLB = 1;
        stateUB = 50;
        actionLB = -1;
        actionUB = 1;
%         actionLB = -inf;
%         actionUB = inf;
        rewardLB = 0;
        rewardUB = 1;

        % Environment variables
        reward_states = [10 41];
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            state = randi(obj.stateUB,1,n);
        end

        function action = parse(obj, action)
            action = min(max(action,obj.actionLB),obj.actionUB);
            noise = rand(size(action));
            action(noise < 0.1) = -action(noise < 0.1);
        end

        function nextstate = transition(obj, state, action)
            nextstate = state + action;
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
        function reward = reward(obj, state, action, nextstate)
            reward = -min(abs(bsxfun(@minus,nextstate,obj.reward_states'))).^2;
        end
        
        function absorb = isterminal(obj, nextstate)
            absorb = false(1,size(nextstate,2));
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            axis off
            
            plot([obj.stateLB obj.stateUB],[1 1],...
                '-b','MarkerSize',10,'MarkerEdgeColor','b','LineWidth',2,'MarkerFaceColor','b')
            plot(obj.reward_states,ones(length(obj.reward_states)),...
                'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
            obj.handleAgent = plot(1,1,...
                'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state(1);
            drawnow limitrate
        end
        
    end
    
end