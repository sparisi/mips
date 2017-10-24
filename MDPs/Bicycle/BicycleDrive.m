classdef BicycleDrive < BicycleEnv
% REFERENCE
% D Ernst, P Geurts, L Wehenkel
% Tree-Based Batch Mode Reinforcement Learning (2005)
    
    properties
        % MDP variables
        dstate = 9;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.98; % 0.98 Ernst, 0.99 Randlov

        % Bounds : state = (theta, theta_dot, omega, omega_dot, psi, xf, yf, xb, yb)
        stateLB = [-1.3963 -inf -pi/15 -inf -pi -inf -inf -inf -inf]';
        stateUB = [1.3963 inf pi/15 inf pi inf inf inf inf]';
        rewardLB = -1;
        rewardUB = 0.002 + 0.01;
        actionLB = 1;
        actionUB = 9;

        allactions = [-2    -2 -2     0    0 0     2    2 2
                      -0.02  0  0.02 -0.02 0 0.02 -0.02 0 0.02];

        goal = [0; 1000];
        goalradius = 10;
    end
    
    methods
        function state = init(obj, n)
            state = repmat([0 0 0 0 0 0 obj.l 0 0]',1,n);
        end
        
        function action = parse(obj, action)
            action = obj.allactions(:,action);
            noise = (rand(1,size(action,2)) * 2 - 1) * obj.maxNoise; % Noise on displacement
            action(2,:) = action(2,:) + noise;
        end
        
        function absorb = isterminal(obj, state)
            omega = state(3,:);
            xf = state(6,:);
            yf = state(7,:);
            isfallen = ( omega < obj.stateLB(3) ) | ( omega > obj.stateUB(3) );            
            dist_goal = matrixnorms(bsxfun(@minus,[xf;yf],obj.goal),2);
            isgoal = dist_goal < obj.goalradius;
            absorb = false(1,size(state,2));
            absorb(isfallen | isgoal) = true;
        end
        
        function reward = reward(obj, state, action, nextstate)
            omega_next = nextstate(3,:);
            xf = nextstate(6,:);
            yf = nextstate(7,:);
            isfallen = ( omega_next < obj.stateLB(3) ) | ( omega_next > obj.stateUB(3) );            
            dist_goal = matrixnorms(bsxfun(@minus,[xf;yf],obj.goal),2);
            isgoal = dist_goal < obj.goalradius;

%             reward = obj.rewardRandlov(nextstate);
            reward = obj.rewardErnst(state,nextstate);
            reward(isfallen) = -1;
            reward(isgoal & ~isfallen) = 0.01;
        end
    end
    
end