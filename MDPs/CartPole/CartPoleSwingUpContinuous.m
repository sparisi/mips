classdef CartPoleSwingUpContinuous < CartPoleEnv
% Cart-pole with continuous actions.
% The goal is to balance the pole starting from a random position.

    properties
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;

        % Bounds : state = [x xd theta thetad])
        stateLB = [-3, -inf, -pi, -inf]';
        stateUB = [3, inf, pi, inf]';
        actionLB = -CartPoleEnv.force;
        actionUB = CartPoleEnv.force;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods
        function state = init(obj, n)
            state = myunifrnd([-1,-2,pi-1,-3],[1,2,pi+1,3],n);
        end
        
        function action = parse(obj, action)
            action = max(min(action,obj.actionUB),obj.actionLB);
        end
            
        function reward = reward(obj, state, action, nextstate)
            reward = cos(nextstate(3,:)) - 1;
            reward(obj.isterminal(nextstate)) = -100;
        end
        
        function absorb = isterminal(obj, nextstate)
            absorb = nextstate(1,:) < obj.stateLB(1) | nextstate(1,:) > obj.stateUB(1);
        end
    end
 
end