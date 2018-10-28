classdef CartPoleContinuous < CartPoleEnv
% Cart-pole with continuous actions.
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
        actionLB = -CartPoleEnv.force;
        actionUB = CartPoleEnv.force;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods
        function state = init(obj, n)
            state = zeros(obj.dstate,n);
            theta = myunifrnd(-deg2rad(5),deg2rad(5),n);
            theta = wrapinpi(theta);
            state(3,:) = theta;
        end
        
        function action = parse(obj, action)
            action = max(min(action,obj.actionUB),obj.actionLB);
        end
            
        function reward = reward(obj, state, action, nextstate)
            reward = zeros(1,size(state,2));
            reward(obj.isterminal(state)) = -1;
        end
        
        function absorb = isterminal(obj, state)
            absorb = sum(bsxfun(@lt, state, obj.stateLB),1) | ...
                sum(bsxfun(@gt, state, obj.stateUB),1);
        end
    end
 
end