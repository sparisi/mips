classdef DeepSeaTreasure < MOMDP
% REFERENCE
% P Vamplew, R Dazeley, A Berry, R Issabekov, E Dekker
% Empirical evaluation methods for multiobjective reinforcement learning 
% algorithms (2011) 
    
    %% Properties
    properties
        % Environment variables
        treasure = ...
           [0   0   0   0   0   0   0   0   0   0
            1   0   0   0   0   0   0   0   0   0
            0   2   0   0   0   0   0   0   0   0
            0   0   3   0   0   0   0   0   0   0
            0   0   0   5   8  16   0   0   0   0
            0   0   0   0   0   0   0   0   0   0
            0   0   0   0   0   0   0   0   0   0
            0   0   0   0   0   0  24  50   0   0
            0   0   0   0   0   0   0   0   0   0
            0   0   0   0   0   0   0   0  74   0
            0   0   0   0   0   0   0   0   0 124];
        isopen = ...
           [1   1   1   1   1   1   1   1   1   1
            1   1   1   1   1   1   1   1   1   1
            0   1   1   1   1   1   1   1   1   1
            0   0   1   1   1   1   1   1   1   1
            0   0   0   1   1   1   1   1   1   1
            0   0   0   0   0   0   1   1   1   1
            0   0   0   0   0   0   1   1   1   1
            0   0   0   0   0   0   1   1   1   1
            0   0   0   0   0   0   0   0   1   1
            0   0   0   0   0   0   0   0   1   1
            0   0   0   0   0   0   0   0   0   1];
        
        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 2;
        isAveraged = 0;
        gamma = 1;
        
        % Bounds
        stateLB = [1 1]';
        stateUB = [11 10]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = [0 -inf]';
        rewardUB = [124 -1]';
        
        % Multiobjective
        utopia = [124 -1];
        antiutopia = [0 -20];
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj, n)
            state = repmat([1; 1],1,n);

%             [d1,d2] = size(obj.treasure);
%             [S1, S2] = meshgrid(1:d1,1:d2);
%             S = [S1(:) S2(:)]';
%             idx1 = logical(obj.isopen(d1*(S(2,:)-1) + S(1,:)));
%             istreasure = obj.treasure > 0;
%             idx2 = ~logical(istreasure(d1*(S(2,:)-1) + S(1,:)));
%             S = S(:, idx1 & idx2);
%             state = S(:, randi(size(S,2),1,n));

            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstates = size(state,2);
            
            steps = [0  0  -1  1
                     -1 1   0  0]; % action mapping (left right up down)
            nextstate = state + steps(:,action);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(:,~isopen) = state(:,~isopen);
            
            % Reward function
            reward1 = obj.treasure(size(obj.treasure,1)*(nextstate(2,:)-1) + nextstate(1,:)); % Treasure value
            reward2 = -ones(1,nstates); % Time
            reward = [reward1; reward2];

            absorb = reward1 > 0;
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
        %% Multiobjective
        function [front, weights] = truefront(obj)
            front = dlmread('deep_front.dat');
            weights = [];
        end

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(front, varargin{:});
            xlabel 'Treasure'
            ylabel 'Time'
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.isopen;
            treasurecells = obj.treasure;
            cells(obj.treasure > 0) = 0.5;
            
            cells = flipud(cells);
            h = image(cells); % Plot environment
            colormap([0 0 0; 0.5 0.5 0.5; 1 1 1]);
            
            imggrid(h,'k',0.5); % Add grid
            
            treasurecells = flipud(treasurecells)'; % Cartesian coord -> Matrix coord
            [rows,cols] = find(treasurecells);
            for i = 1 : length(rows) % Add treasures value
                text('position', [rows(i) cols(i)], ...
                    'fontsize', 10, ...
                    'string', num2str(treasurecells(rows(i),cols(i))), ...
                    'color', 'white', ...
                    'horizontalalignment', 'center')
            end
            axis off
            
            obj.handleAgent = plot(1,11,'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            % Convert coordinates from cartesian to matrix
            nrows = size(obj.treasure);
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