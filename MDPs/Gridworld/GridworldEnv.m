classdef GridworldEnv < MDP
% Gridworld with sparse reward (0 everywhere, except for few cells).
    
    %% Properties
    properties
        % Finite states and actions
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

        probT = [1, 0, 0];
        % probT = [a, b, c]
        % a is the chance to do the correct action
        % b is the chance to do a random action
        % c is the chance to stay in the same state
        % a+b+c = 1

        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        
        % Bounds
        actionLB = 1;
        actionUB = 4;
    end
    
    methods
        
        function add_noise(obj, probT)
            if nargin == 1, probT = [0.95, 0.025, 0.025]; end
            assert(sum(probT) == 1, 'Transition probablities do not sum to 1.')
            obj.probT = probT;
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

            % Any reward or penalty cell is terminal
            absorb = reward~=0;
            
            % Add a small constant penalty
            reward = reward - 0.01;
            
            if obj.realtimeplot, obj.updateplot(state), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.reward;
            cells2 = cells;
            cells2(~obj.isopen) = -100;
            h = image(flipud(cells2)); % Plot environment

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

            % Rotate and center
            camroll(90)
            pos = get(obj.handleEnv.Children, 'Position');
            pos(1) = pos(1) - 0.07;
            pos(2) = pos(2) + 0.05;
            set(obj.handleEnv.Children, 'Position', pos)
            set(obj.handleEnv.CurrentAxes.Title, 'String', '')
            set(obj.handleEnv.CurrentAxes.Title, 'Position', ...
                get(obj.handleEnv.CurrentAxes.Title, 'Position') + [-8 -5 0])
        end
        
        function updateplot(obj, state)
            [obj.handleAgent.XData, obj.handleAgent.YData] = ...
                cart2mat(state(1),state(2),size(obj.reward,1));
            drawnow limitrate
        end
        
    end
    
end