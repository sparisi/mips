classdef Resource < MOMDP
% REFERENCE
% P Vamplew, R Dazeley, A Berry, R Issabekov, E Dekker
% Empirical evaluation methods for multiobjective reinforcement learning 
% algorithms (2011) 
    
    %% Properties
    properties
        % Environment variables
        gems = [0 0 0 0 0
            0 0 0 0 1 
            0 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0];
        gold = [0 0 1 0 0
            0 0 0 0 0 
            0 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0];
        enemy = [0 0 0 1 0
            0 0 1 0 0
            0 0 0 0 0
            0 0 0 0 0
            0 0 0 0 0];
        
        % Finite states and actions
        allstates = allcomb([1 2 3 4 5], [1 2 3 4 5], [0 1], [0 1]);
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down
        
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 3;
        isAveraged = 1;
        gamma = 1;
        
        % Bounds
        stateLB = [1 1 0 0]';
        stateUB = [5 5 1 1]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = [0 0 -1]';
        rewardUB = [1 1 0]';
        
        % Multiobjective
        utopia = [0 0.1120 0.1000];
        antiutopia = [-0.0263 0 0];
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            state = repmat([5 3 0 0]',1,n);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nextstate = state;
            nextstate(1:2,:) = state(1:2,:) + obj.allactions(:,action);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Check gems and gold
            hasGems = logical(obj.gems(size(obj.gems,1)*(nextstate(2,:)-1) + nextstate(1,:)));
            nextstate(3,hasGems) = 1;
            hasGold = logical(obj.gold(size(obj.gold,1)*(nextstate(2,:)-1) + nextstate(1,:)));
            nextstate(4,hasGold) = 1;
            
            % Check for enemies
            hasEnemy = obj.enemy(size(obj.enemy,1)*(nextstate(2,:)-1) + nextstate(1,:));
            fightLost = rand(1,size(state,2)) < 0.1;
            backHome = fightLost & hasEnemy;
            if sum(backHome) > 0
                nextstate(:,backHome) = obj.initstate(sum(backHome));
            end

            % Reward function
            reward = zeros(obj.dreward,size(state,2));
            homeGems = (nextstate(1,:) == 5 & nextstate(2,:) == 3 & nextstate(3,:) == 1);
            homeGold = (nextstate(1,:) == 5 & nextstate(2,:) == 3 & nextstate(4,:) == 1);
            reward(1,homeGems) = 1; % Bonus for carrying gems home
            reward(2,homeGold) = 1; % Bonus for carrying gold home
            reward(3,backHome) = -1; % Penalty for losing a fight
            
            nextstate(3,homeGems) = 0; % Reset gems
            nextstate(4,homeGold) = 0; % Reset gold
            
            absorb = false(1,size(state,2)); % Infinite horizon
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
        %% Multiobjective
        function [front, weights] = truefront(obj)
            front = dlmread('resource_front.dat');
            weights = [];
        end

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(front, varargin{:});
            xlabel 'Fight Penalty'
            ylabel 'Gold'
            zlabel 'Gems'
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all

            cells = zeros(size(obj.gems));
            cells(5,3) = 1;
            
            hasGold = flipud(obj.gold)'; % Cartesian coord -> Matrix coord
            hasGems = flipud(obj.gems)';
            hasEnemy = flipud(obj.enemy)';
            
            h = imagesc(flipud(cells)); % Plot cells
            imggrid(h,'k',0.5); % Add grid
            colormap([1 1 1; 0.7 0.7 0.7])
            
            [x,y] = find(hasGold);
            obj.handleAgent{2} = plot(x,y,'ko','MarkerSize',24,'MarkerFaceColor','y'); % Gold
            
            [x,y] = find(hasGems);
            obj.handleAgent{3} = plot(x,y,'kdiamond','MarkerSize',24,'MarkerFaceColor','m'); % Gems
            
            [x,y] = find(hasEnemy);
            plot(x, y, 'kx', 'MarkerSize', 35, 'LineWidth', 15); % Enemies
            
            axis off
            
            obj.handleAgent{1} = plot(3,1,'ro','MarkerSize',10,'MarkerFaceColor','r'); % Agent
        end
        
        function updateplot(obj, state)
            hasGold = flipud(obj.gold)';
            hasGems = flipud(obj.gems)';
            [xs, ys] = cart2mat(state(1),state(2),size(obj.gems,1));
            obj.handleAgent{1}.XData = xs;
            obj.handleAgent{1}.YData = ys;
            
            if state(3) % Carrying gems
                obj.handleAgent{2}.XData = xs+0.3;
                obj.handleAgent{2}.YData = ys+0.3;
                obj.handleAgent{2}.MarkerSize = 5;
            else
                [x,y] = find(hasGems);
                obj.handleAgent{2}.XData = x;
                obj.handleAgent{2}.YData = y;
                obj.handleAgent{2}.MarkerSize = 24;
            end
            
            if state(4) % Carrying gold
                obj.handleAgent{3}.XData = xs+0.3;
                obj.handleAgent{3}.YData = ys+0.1;
                obj.handleAgent{3}.MarkerSize = 5;
            else
                [x,y] = find(hasGold);
                obj.handleAgent{3}.XData = x;
                obj.handleAgent{3}.YData = y;
                obj.handleAgent{3}.MarkerSize = 24;
            end
            
            drawnow limitrate
        end
        
    end
    
end