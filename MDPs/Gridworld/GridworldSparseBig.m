classdef GridworldSparseBig < GridworldEnv
% Bigger grid and more sparse rewards.
    
    %% Properties
    properties
        % Environment variables
        reward = zeros(100,100);
        isopen = ones(100,100);
        
        % Finite states and actions
        allstates = allcomb(1:100, 1:100);
        
        % Bounds
        stateLB = [1 1]';
        stateUB = [100 100]';
        rewardLB
        rewardUB
    end
    
    methods
        
        %% Constructor
        function obj = GridworldSparseBig()
            obj.reward(100,100) = 1000;
            obj.reward(10,22) = 15;
            obj.reward(22,24) = 3;
            obj.reward(31,8) = 22;
            obj.reward(30,30) = -20;
            obj.reward(40,20) = -15;
            obj.reward(20,40) = -15;
            obj.reward(30,60) = 45;
            obj.reward(35,63) = -45;
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