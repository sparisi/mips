classdef Puddleworld < PuddleworldEnv
    
    %% Properties
    properties
        % Finite actions
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds
        stateLB = [0 0]';
        stateUB = [1 1]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = -41;
        rewardUB = -1;
    end
    
    methods
        function state = init(obj, n)
            state = rand(2,n);
        end
        
        function action = parse(obj, action)
            action = obj.allactions(:,action) * obj.step;
        end
            
        function reward = reward(obj, state, action, nextstate)
            reward = obj.puddlepenalty(nextstate) - 1;
        end
        
        function absorb = isterminal(obj, state)
            absorb = sum(state,1) >= 1.9;
        end
    end
    
end
