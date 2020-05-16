classdef GridworldMO < MOMDP
% Simple NxN gridworld with rewards at the bottom row. 
% The reward in the middle has the smallest value. The further from the 
% middle the reward is, the higher its value. 
% The agent always starts in the middle of the top row.
    
    %% Properties
    properties
        N % Size of the grid
        
        % Finite states and actions
        allactions = [0  0  -1  1
                     -1 1   0  0]; % Left right up down

        % MDP variables
        dstate = 2; % [x, y]
        daction = 1;
        dreward = 2;
        isAveraged = 0;
        gamma = 1;
        
        % Finite states and actions
        allstates
        
        % Reward matrix
        reward

        % Bounds
        stateLB
        stateUB
        actionLB = 1;
        actionUB = 4;
        rewardLB = [0 -inf]';
        rewardUB = [inf 0]';

        % Multiobjective
        utopia = [inf -1];
        antiutopia = [0 -inf];
    end
    
    methods
        
        function obj = GridworldMO(N)
            if nargin == 0; N = 8; end
            obj.N = N;
            obj.rewardUB(1) = floor(N/2);
            obj.allstates = allcomb(1:N, 1:N);
            obj.stateLB = [1 1]';
            obj.stateUB = [N N]';
            
            x = 0.1;
            
            obj.utopia = [floor(N/2) -floor(N)+1];
            obj.antiutopia = [1 -N-floor(N/2)+1];
            if mod(obj.N,2) == 0
                obj.antiutopia(2) = obj.antiutopia(2) + 1;
            else
                obj.antiutopia(1) = x;
            end
            
            obj.utopia = obj.utopia + 0.1;
            obj.antiutopia = obj.antiutopia - 0.1;

            obj.reward = zeros(N);
            r = [floor(N/2):-1:1, 1:floor(N)/2];
            if mod(N,2) > 0
                r = [r(1:floor(N/2)), x, r(floor(N/2)+1:end)];
            end
            obj.reward(N,:) = r;
        end

        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = [ones(1,n); ceil(obj.N/2)*ones(1,n)];
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nextstate = state;
            nextstate(1:2,:) = state(1:2,:) + obj.allactions(:,action);
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Reward function
            reward = obj.reward(size(obj.reward,1)*(nextstate(2,:)-1) + nextstate(1,:));

            % Any reward cell is terminal
            absorb = reward~=0;
            
            reward = [reward; -ones(1,size(reward,2))];

            if obj.realtimeplot, obj.updateplot(state), end
        end

        %% Multiobjective
        function [front, weights] = truefront(obj)
            if mod(obj.N,2) > 0
                front = [obj.reward(end,1:ceil(obj.N/2))', [-obj.N-floor(obj.N/2)+1:-floor(obj.N)+1]'];
            else
                front = [obj.reward(end,1:ceil(obj.N/2))', [-obj.N-floor(obj.N/2)+1:-floor(obj.N)]'];
            end
            if mod(obj.N,2) == 0
                front(:,2) = front(:,2) + 1;
            end
            weights = [];
        end

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(front, varargin{:});
            xlabel 'Obj 1'
            ylabel 'Obj 2'
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