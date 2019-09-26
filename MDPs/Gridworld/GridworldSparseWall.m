classdef GridworldSparseWall < MDP
% Like GridworldSparse, but with a wall which makes it harder to reach the
% highest reward.
    
    %% Properties
    properties
        % Environment variables
        reward = zeros(50,50);
        isopen = ones(50,50);
        
        probT = [1, 0, 0];
%         probT = [0.8, 0.15, 0.05];

        % probT = [a, b, c]
        % a is the chance to do the correct action
        % b is the chance to do a random action
        % c is the chance to stay in the same state
        % a+b+c = 1
        
        % Finite states and actions
        allstates = allcomb(1:50, 1:50);
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        
        % Bounds
        stateLB = [1 1]';
        stateUB = [50 50]';
        actionLB = 1;
        actionUB = 4;
        rewardLB
        rewardUB
    end
    
    methods
        
        %% Constructor
        function obj = GridworldSparseWall()
            obj.reward(50,50) = 500;
            obj.reward(10,22) = 15;
            obj.reward(22,24) = 3;
            obj.reward(31,8) = 22;
            obj.reward(30,30) = -20;
            obj.reward(38,20) = -15;
            obj.reward(20,38) = -15;
            obj.rewardLB = min(obj.reward(:));
            obj.rewardUB = max(obj.reward(:));
            
            obj.isopen(40,1:40) = 0;
            obj.isopen(5:40,40) = 0;
        end
        
        %% Simulator
        function state = init(obj, n)
            state = 25 * ones(2,n);
%             state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
        end
        
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
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end

    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.reward;
            cells(~obj.isopen) = -100;
            h = image(flipud(cells)); % Plot environment

            imggrid(h,'k',0.5); % Add grid

            cells = flipud(cells)';
            [rows,cols] = find(cells);
            for i = 1 : length(rows) % Add value
                if ~ (cells(rows(i),cols(i)) == -100)
                    text('position', [rows(i) cols(i)], ...
                        'fontsize', 10, ...
                        'string', num2str(cells(rows(i),cols(i))), ...
                        'color', 'red', ...
                        'horizontalalignment', 'center')
                end
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