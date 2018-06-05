classdef CartPole < CartPoleEnv
% Cart-pole with discrete actions.
% The goal is to balance the pole starting from upright position.
% The episode ends if the pole tilts too much or if the cart goes out of
% bounds.
    
    properties
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.9;

        % Bounds
        stateLB = [-2.4, -inf, -deg2rad(15), -inf]';
        stateUB = [2.4, inf, deg2rad(15), inf]';
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 0;

        % Finite actions
        allactions = [-CartPoleEnv.force CartPoleEnv.force];
    end
    
    methods
        function state = init(obj, n)
            state = zeros(obj.dstate,n);
            state(3,:) = myunifrnd(-deg2rad(15)*0.05,deg2rad(15)*0.05,n);
        end
        
        function action = parse(obj, action)
            action = obj.allactions(action);
        end
            
        function reward = reward(obj, state, action, nextstate)
            reward = zeros(1,size(nextstate,2));
            reward(obj.isterminal(nextstate)) = -1;
        end
        
        function absorb = isterminal(obj, state)
            absorb = sum(bsxfun(@lt, state, obj.stateLB),1) | ...
                sum(bsxfun(@gt, state, obj.stateUB),1);
        end
    end
 
end