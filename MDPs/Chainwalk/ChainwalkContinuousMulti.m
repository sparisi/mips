classdef ChainwalkContinuousMulti < MDP
    
    %% Properties
    properties
        % MDP variables
        dstate
        daction
        dreward = 1;
        isAveraged = 0;
        gamma = 1;
        
        % Bounds
        stateLB
        stateUB
        actionLB
        actionUB
        rewardLB = 0;
        rewardUB = 1;

        % Environment variables
        reward_states;
    end
    
    methods
        
        function obj = ChainwalkContinuousMulti(n)
            obj.dstate = n;
            obj.daction = n;
            obj.stateLB = ones(n,1);
            obj.stateUB = 50*ones(n,1);
            obj.actionLB = -ones(n,1);
            obj.actionUB = ones(n,1);
            obj.reward_states = [10 50 15
                                 11 49 14];
        end            
        
        %% Simulator
        function state = init(obj, n)
            state = randi([obj.stateLB(1), obj.stateUB(1)], ...
                length(obj.stateLB), n);
        end

        function action = parse(obj, action)
            action = bsxfun(@min, bsxfun(@max, action, obj.actionLB), obj.actionUB);
            noise = rand(size(action));
            action(noise < 0.1) = -action(noise < 0.1);
        end

        function nextstate = transition(obj, state, action)
            nextstate = state + action;
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
        function reward = reward(obj, state, action, nextstate)
            reward = -min(sqrt(sum(bsxfun(@minus,nextstate',permute(obj.reward_states,[3 1 2])).^2,2)),[],3)';
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
            plot([obj.stateLB(1) obj.stateLB(1) obj.stateUB(1) obj.stateUB(1) obj.stateLB(1)], ...
                [obj.stateLB(2) obj.stateUB(2) obj.stateUB(2) obj.stateLB(2) obj.stateLB(2)],...
                '-b','MarkerSize',10,'MarkerEdgeColor','b','LineWidth',2,'MarkerFaceColor','b')
            plot(obj.reward_states(1,:), obj.reward_states(2,:),...
                'og','MarkerSize',10,'MarkerEdgeColor','g','LineWidth',2,'MarkerFaceColor','g')
            obj.handleAgent = plot(-1,-1,...
                'ro','MarkerSize',8,'MarkerFaceColor','r');
            axis([obj.stateLB(1) obj.stateUB(1) obj.stateLB(2) obj.stateUB(2)])
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state(1,1);
            obj.handleAgent.YData = state(2,1);
            drawnow limitrate
        end
        
    end
    
end