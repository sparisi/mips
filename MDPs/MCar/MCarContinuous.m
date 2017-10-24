classdef MCarContinuous < MCarEnv

    properties
        % MDP Variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        gamma = 0.95;
        isAveraged = 0;

        % Bounds
        stateLB = [-1 -3]';
        stateUB = [1 3]';
        actionLB = -4;
        actionUB = 4;
        rewardLB = -1;
        rewardUB = 1;
    end
    
    methods
        function state = init(obj, n)
            state = repmat([-0.5, 0]',1,n);
            state = [myunifrnd(-0.6,-0.4,n); zeros(1,n)];
        end
        
        function action = parse(obj, action)
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
        end
    end
    
end
