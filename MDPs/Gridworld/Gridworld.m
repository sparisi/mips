classdef Gridworld < MDP
    
    %% Properties
    properties
        % Environment variables
        reward = [ ...
            0   0   0   0   0   9   0
            1   0   0   0   0   0   0
            0   0   0   0   0   0   0
            0   0   0   0   0   0   0
            -9  -9  0   0   -9  0   -9
            0   13 0   0   0   0   0
            0   0   0   0   0   0   0
            ] * 1e1;
        isopen = [ ...
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            1  1  1  1  1  1  1
            ];
        
        % Finite states and actions
        allstates = allcomb([1 2 3 4 5 6 7], [1 2 3 4 5 6 7]);
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

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
        function state = init(obj, n)
            state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nextstate = state + obj.allactions(:,action);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(:,~isopen) = state(:,~isopen);
            
            % Reward function
            reward = obj.reward(size(obj.reward,1)*(nextstate(2,:)-1) + nextstate(1,:));

            % Any reward or penalty cell is terminal
            absorb = reward~=0;
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.reward;
            h = image(flipud(cells)); % Plot environment

            imggrid(h,'k',0.5); % Add grid

            cells = flipud(cells)';
            [rows,cols] = find(cells);
            for i = 1 : length(rows) % Add value
                text('position', [rows(i) cols(i)], ...
                    'fontsize', 10, ...
                    'string', num2str(cells(rows(i),cols(i))), ...
                    'color', 'red', ...
                    'horizontalalignment', 'center')
            end
            
            axis off
            
            obj.handleAgent = plot(1,7,'bo','MarkerSize',8,'MarkerFaceColor','w');
        end
        
        function updateplot(obj, state)
            [obj.handleAgent.XData, obj.handleAgent.YData] = ...
                cart2mat(state(1),state(2),size(obj.reward,1));
            drawnow limitrate
        end
        
    end
    
end