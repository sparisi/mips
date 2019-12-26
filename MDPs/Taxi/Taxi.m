classdef Taxi < MDP
% Gridworld with sparse reward (0 everywhere, except for few cells).
    
    %% Properties
    properties
        % Environment variables
        isopen = [1 0 1 1 0 1 1
            1 0 1 1 0 1 1
            1 1 1 1 1 1 1
            0 0 1 1 1 0 0 
            1 1 1 1 1 1 1
            1 1 1 1 1 1 0];
        
        dest = [1,7];
        start = [1,1];
        flag = [1,3; 5,7; 6,1];
        
        probT = [1, 0, 0];
%         probT = [0.8, 0.15, 0.05];

        % probT = [a, b, c]
        % a is the chance to do the correct action
        % b is the chance to do a random action
        % c is the chance to stay in the same state
        % a+b+c = 1
        
        % Finite states and actions
        allstates = allcomb(1:6, 1:7, 0:1, 0:1, 0:1);
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

        % MDP variables
        dstate = 5;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        
        % Bounds
        stateLB = [1 1 0 0 0]';
        stateUB = [6 7 1 1 1]';
        actionLB = 1;
        actionUB = 4;
        rewardLB = 0;
        rewardUB = 15;
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = ones(2,n);
            state = [state; zeros(3,n)];
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            r = rand(1,size(state,2));
            wrong = r >= obj.probT(1) & r < (obj.probT(1) + obj.probT(2));
            stay = r >= (obj.probT(1) + obj.probT(2));
            action(wrong) = randi(obj.actionUB,1,sum(wrong));
            
            nextstate = state(1:2,:) + obj.allactions(:,action);
            nextstate(:,stay) = state(:,stay);
            
            % Bound the state
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB(1:2)), obj.stateLB(1:2));
            
            % Check if the cell is not black
            isopen = obj.isopen(size(obj.isopen,1)*(nextstate(2,:)-1) + nextstate(1,:));
            nextstate(1:2,~isopen) = state(1:2,~isopen);
            
            % Check pickup
            had_p1 = state(3,:);
            had_p2 = state(4,:);
            had_p3 = state(5,:);
            has_p1 = (nextstate(1,:) == obj.flag(1,1)) & (nextstate(2,:) == obj.flag(1,2));
            has_p2 = (nextstate(1,:) == obj.flag(2,1)) & (nextstate(2,:) == obj.flag(2,2));
            has_p3 = (nextstate(1,:) == obj.flag(3,1)) & (nextstate(2,:) == obj.flag(3,2));
            has_p1 = has_p1 | had_p1;
            has_p2 = has_p2 | had_p2;
            has_p3 = has_p3 | had_p3;
            nextstate(3,:) = has_p1;
            nextstate(4,:) = has_p2;
            nextstate(5,:) = has_p3;
            
            % Check if absorb
            absorb = (state(1,:) == obj.dest(1)) & (state(2,:) == obj.dest(2));

            % Reward function
            n_p = has_p1 + has_p2 + has_p3;
            reward = zeros(size(absorb));
            reward(n_p==1) = 1;
            reward(n_p==2) = 3;
            reward(n_p==3) = 15;
            reward(~absorb) = 0;

            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all

            cells = obj.isopen;
            cells(cells==0) = -100;
            
            h = imagesc(flipud(cells)); % Plot cells
            imggrid(h,'k',0.5); % Add grid
            colormap([0 0 0; 0.5 0.5 0.5; 1, 1, 1])
            
            xy = [obj.flag(:,2),7-obj.flag(:,1)];
            obj.handleAgent{2} = plot(xy(1,1),xy(1,2),'ko','MarkerSize',24,'MarkerFaceColor','y');
            obj.handleAgent{3} = plot(xy(2,1),xy(2,2),'ko','MarkerSize',24,'MarkerFaceColor','m');
            obj.handleAgent{4} = plot(xy(3,1),xy(3,2),'ko','MarkerSize',24,'MarkerFaceColor','b');
            
            axis off
            
            text('position', [obj.start(:,2),7-obj.start(:,1)]+0.3, ...
                'fontsize', 20, ...
                'string', 'S', ...
                'color', 'k', ...
                'horizontalalignment', 'center')
            text('position', [obj.dest(:,2),7-obj.dest(:,1)]+0.3, ...
                'fontsize', 20, ...
                'string', 'D', ...
                'color', 'k', ...
                'horizontalalignment', 'center')
            
            obj.handleAgent{1} = plot(obj.start(2),7-obj.start(1),'ro','MarkerSize',10,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            [xs, ys] = cart2mat(state(1),state(2),size(obj.isopen,1));
            obj.handleAgent{1}.XData = xs;
            obj.handleAgent{1}.YData = ys;
            
            xy = [obj.flag(:,2),7-obj.flag(:,1)];
            if state(3)
                obj.handleAgent{2}.XData = xs+0.3;
                obj.handleAgent{2}.YData = ys+0.3;
                obj.handleAgent{2}.MarkerSize = 5;
            else
                obj.handleAgent{2}.XData = xy(1,1);
                obj.handleAgent{2}.YData = xy(1,2);
                obj.handleAgent{2}.MarkerSize = 24;
            end
            
            if state(4)
                obj.handleAgent{3}.XData = xs+0.3;
                obj.handleAgent{3}.YData = ys+0.1;
                obj.handleAgent{3}.MarkerSize = 5;
            else
                obj.handleAgent{3}.XData = xy(2,1);
                obj.handleAgent{3}.YData = xy(2,2);
                obj.handleAgent{3}.MarkerSize = 24;
            end
            
            if state(5)
                obj.handleAgent{4}.XData = xs+0.1;
                obj.handleAgent{4}.YData = ys+0.3;
                obj.handleAgent{4}.MarkerSize = 5;
            else
                obj.handleAgent{4}.XData = xy(3,1);
                obj.handleAgent{4}.YData = xy(3,2);
                obj.handleAgent{4}.MarkerSize = 24;
            end
            
            drawnow limitrate
        end
        
    end
    
end