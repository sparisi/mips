classdef BicycleBalance < BicycleEnv
    
    properties
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;

        % Bounds : state = (theta, theta_dot, omega, omega_dot)
        stateLB = [-1.3963 -inf -pi/15 -inf]';
        stateUB = [1.3963 inf pi/15 inf]';
        rewardLB = -1;
        rewardUB = 1;
        actionLB = 1;
        actionUB = 9;

        allactions = [-2    -2 -2     0    0 0     2    2 2
                      -0.02  0  0.02 -0.02 0 0.02 -0.02 0 0.02];
    end
    
    methods
        function state = init(obj, n)
            state = repmat([0 0 0 0]',1,n);
        end
        
        function action = parse(obj, action)
            action = obj.allactions(:,action);
            noise = (rand(1,size(action,2)) * 2 - 1) * obj.maxNoise; % Noise on displacement
            action(2,:) = action(2,:) + noise;
        end
        
        function reward = reward(obj, state, action, nextstate)
            omega_s = state(3,:);
            omega_next = nextstate(3,:);
            reward = (omega_s / obj.stateLB(3)).^2 - (omega_next / obj.stateLB(3)).^2;
        end
        
        function absorb = isterminal(obj, nextstate)
            omega_next = nextstate(3,:);
            isfallen = ( omega_next < obj.stateLB(3) ) | ( omega_next > obj.stateUB(3) );            
            absorb = false(1,size(nextstate,2));
            absorb(isfallen) = true;
        end
    end
    
end