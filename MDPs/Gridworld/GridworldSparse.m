classdef GridworldSparse < GridworldEnv
% Gridworld with sparse reward (0 everywhere, except for few cells).
    
    %% Properties
    properties
        % Environment variables
        reward = zeros(50,50);
        isopen = ones(50,50);
        
        % Finite states and actions
        allstates = allcomb(1:50, 1:50);

        % Bounds
        stateLB = [1 1]';
        stateUB = [50 50]';
        rewardLB
        rewardUB
    end
    
    methods
        
        %% Constructor
        function obj = GridworldSparse()
            obj.reward(50,50) = 50;
            obj.reward(10,22) = 15;
            obj.reward(22,24) = 3;
            obj.reward(31,8) = 22;
            obj.reward(30,30) = -20;
            obj.reward(40,20) = -15;
            obj.reward(20,40) = -15;
            obj.rewardLB = min(obj.reward(:));
            obj.rewardUB = max(obj.reward(:));
        end
        
        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = 25 * ones(2,n);
%             state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
        end
        
    end
    
end