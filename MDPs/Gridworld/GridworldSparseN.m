classdef GridworldSparseN < GridworldEnv
% NxN with rewards on the diagonal. 
% No state is terminal.
% Added "stand still" action. 
% Optimal policy is to navigate through the diagonal and then stand in the last cell.
    
    %% Properties
    properties
        stateLB
        stateUB
        N
        reward
        isopen
        allstates
        rewardLB = 0;
        rewardUB = 1;
    end
    
    methods
        
        %% Constructor
        function obj = GridworldSparseN(N)
            obj.N = N;
            obj.reward = zeros(N);
            obj.reward(N,N) = 1;
            obj.reward(1,1) = 1/N^2;
            obj.isopen = ones(N);

            obj.allstates = allcomb(1:N, 1:N);

            obj.stateLB = [1 1]';
            obj.stateUB = [N N]';

            obj.allactions = [0  0  -1  1 0
                         -1 1   0  0 0]; % Left right up down stand
            obj.actionUB = 5;
        end
        
        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = 1 * ones(2,n);
%             state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
        end
        
        %% Simulator
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            r = rand(1,size(state,2));
            wrong = r >= obj.probT(1) & r < (obj.probT(1) + obj.probT(2));
            stay = r >= (obj.probT(1) + obj.probT(2));
            action(wrong) = randi(obj.actionUB,1,sum(wrong));
            
            nextstate = state + obj.allactions(:,action);
            nextstate(:,stay) = state(:,stay);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(:,~isopen) = state(:,~isopen);
            
            % Reward function
            reward = obj.reward(size(obj.reward,1)*(state(2,:)-1) + state(1,:));

            % End only at horizon
            absorb = false(1,size(state,2));
            
            if obj.realtimeplot, obj.updateplot(state), end
        end
    end
    
end