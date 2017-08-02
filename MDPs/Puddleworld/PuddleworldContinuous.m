classdef PuddleworldContinuous < PuddleworldEnv
    
    %% Properties
    properties
        % MDP variables
        dstate = 2;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds
        stateLB = [0 0]';
        stateUB = [1 1]';
        actionLB = -[0.05 0.05]';
        actionUB = [0.05 0.05]';
        rewardLB = -41;
        rewardUB = -1;
    end
    
    methods
        function state = init(obj, n)
            state = rand(2,n);
        end
        
        function action = parse(obj, action)
%             idx = matrixnorms(action,2) == 0;
%             action = bsxfun(@times,action,1./matrixnorms(action,2)) * obj.step;
%             action(:,idx) = 0;
            bounded_action = min(max(action,obj.actionLB),obj.actionUB);
            action = bounded_action;
        end
            
        function reward = reward(obj, state, action, nextstate)
            reward = obj.puddlepenalty(nextstate) - 1;
        end
        
        function absorb = isterminal(obj, nextstate)
            absorb = sum(nextstate,1) >= 1.9;
        end
    end
    
end
