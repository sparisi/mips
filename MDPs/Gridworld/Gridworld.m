classdef Gridworld < MDP
    
    %% Properties
    properties
        % Environment variables
        reward = [ ...
            0  0  0  0  0  5  0
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
           -9 -9  0  0 -9  0 -9
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
            ] * 1e1;
        terminal = logical([ ...
            0  0  0  0  0  1  0
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
            1  1  0  0  1  0  1
            0  0  0  0  0  0  0
            0  0  0  0  0  0  0
            ]);
        isopen = [ ...
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            ];
        
        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.8;
        
        % Bounds
        stateLB = [1 1]';
        stateUB = [7 7]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = 0;
        rewardUB = 2;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj, n)
            state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
%             state = repmat([1; 1], 1, n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            steps = [0  0  -1  1
                     -1 1   0  0]; % action mapping (left right up down)
            nextstate = state + steps(:,action);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(:,~isopen) = state(:,~isopen);
            
            % Reward function
            reward = obj.reward(size(obj.reward,1)*(nextstate(2,:)-1) + nextstate(1,:));

            absorb = obj.terminal(size(obj.terminal,1)*(nextstate(2,:)-1) + nextstate(1,:));
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.reward;
            cells = flipud(cells);
            h = image(cells); % Plot environment
            colormap([1 0 0; 1 1 1; 0 1 0]);
            
            imggrid(h,'k',0.5); % Add grid
            axis off
            
            obj.handleAgent = plot(1,7,'bo','MarkerSize',8,'MarkerFaceColor','b');
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
            drawnow limitrate
        end
        
    end
    
end