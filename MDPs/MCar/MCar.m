classdef MCar < MCarEnv
    
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
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 1;

        % Finite actions
        allactions = [-4 4];

        % For discrete MDPs, actions are encoded as integers (e.g., 1 ... 4
        % for left, right, up, down).
        % In this case, actionLB = 1 and actionUB = 4.
    end
    
    methods
        function state = init(obj, n)
            state = repmat([-0.5, 0]',1,n);
            state = [myunifrnd(-0.6,-0.4,n); zeros(1,n)];
        end
        
        function action = parse(obj, action)
            action = obj.allactions(action);
        end
    end
    
end