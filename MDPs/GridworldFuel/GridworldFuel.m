classdef GridworldFuel < MDP
% REFERENCE
% H Hachiya, M Sugiyama
% Feature Selection for Reinforcement Learning: Evaluating Implicit 
% State-Reward Dependency via Conditional Mutual Information (2010)
    
    %% Properties
    properties
        % Environment variables
        reward = ...
           [0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2
            0   0   0   0   0   0   0   0   0   2];
        isopen = ...
           [1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   1   1   1   1   1
            1   1   1   1   1   1   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1
            1   1   1   1   1   0   1   1   1   1];
        
        % MDP variables
        dstate = 3;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.95;
        
        % Bounds
        stateLB = [1 1 1]';
        stateUB = [10 10 20]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = 0;
        rewardUB = 2;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj, n)
            state = [randi(10,1,n); ones(1,n); 20*ones(1,n)];
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            steps = [0  0  -1  1
                     -1 1   0  0]; % action mapping (left right up down)
            nextstate = state(1:2,:) + steps(:,action);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB(1:2)), obj.stateLB(1:2));
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(:,~isopen) = state(1:2,~isopen);
            nextstate = [nextstate; state(3,:)-1];
            
            % Reward function
            reward = obj.reward(size(obj.reward,1)*(nextstate(2,:)-1) + nextstate(1,:));

            absorb = reward > 0 | nextstate(3,:) == 0;
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.isopen;
            cells(obj.reward > 0) = 0.5;
            
            cells = flipud(cells);
            h = image(cells); % Plot environment
            colormap([0 0 0; 0.5 0.5 0.5; 1 1 1]);
            
            imggrid(h,'k',0.5); % Add grid
            axis off
            
            obj.handleAgent = plot(1,10,'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            % Convert coordinates from cartesian to matrix
            nrows = size(obj.reward);
            convertY = -(-nrows:-1); % Cartesian coord -> Matrix coord
            x = state(2); % (X,Y) -> (Y,X)
            y = state(1);
            state = [x; convertY(y)];
            
            obj.handleAgent.XData = state(1);
            obj.handleAgent.YData = state(2);
        end
        
    end
    
end