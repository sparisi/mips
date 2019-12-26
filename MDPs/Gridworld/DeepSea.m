classdef DeepSea < MDP
% http://jmlr.org/papers/volume20/18-339/18-339.pdf
    
    %% Properties
    properties
        N % Size of the grid
        
        % Finite states and actions
        allactions = [1 1   % Left+down
                     -1 1]; % Right+down

        % MDP variables
        dstate = 3; % [x, y, bomb/treasure]
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;
        
        % Finite states and actions
        allstates

        % Bounds
        stateLB
        stateUB
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 1;
    end
    
    methods
        
        function obj = DeepSea(N)
            if nargin == 0; N = 8; end
            obj.N = N;
            obj.allstates = allcomb(1:N, 1:N, 0:1);
            obj.stateLB = [1 1 0]';
            obj.stateUB = [N N 1]';
        end

        %% Simulator
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = [ones(2,n); randi(2,1,n)-1];
            if n > 1 % In case of evaluation over multiple episodes
                n_half = floor(n/2);
                state = [ones(2,n); [zeros(1,n_half) ones(1,n-n_half)]];
            end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nextstate = state;
            nextstate(1:2,:) = state(1:2,:) + obj.allactions(:,action);
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            % Reward function
            on_end = (sum(state(1:2,:),1) == 2 * obj.N);
            is_penalty = (state(1,:) == state(2,:)) & (action == 2);
            is_treasure = state(3,:) == 1;
            is_bomb = state(3,:) == 0;
            reward = zeros(1,size(state,2));
            reward(on_end & is_treasure) = 1;
            reward(on_end & is_bomb) = - 1;
            reward(is_penalty) = reward(is_penalty) - 0.01 / obj.N;
            
            % Out of bound or final row
            absorb = state(1,:) == obj.N;
            
            if obj.realtimeplot, obj.updateplot(state), end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            h = image(zeros(obj.N)); % Plot environment

            imggrid(h,'k',0.5); % Add grid
            
            axis off
            
            obj.handleAgent = plot(1,1, ...
                'bo','MarkerSize',8,'MarkerFaceColor','w');
            obj.handleAgent(2) = plot(obj.N,1, ...
                'gs','MarkerSize',14,'MarkerFaceColor','g'); % Treasure
            obj.handleAgent(3) = plot(obj.N,1, ...
                'rx','MarkerSize',14,'MarkerFaceColor','r','LineWidth',3); % Bomb
        end
        
        function updateplot(obj, state)
            [obj.handleAgent(1).XData, obj.handleAgent(1).YData] = ...
                cart2mat(state(1),state(2),obj.N);
            if state(3) == 0
                obj.handleAgent(3).set('Visible',1)
                obj.handleAgent(2).set('Visible',0)
            else
                obj.handleAgent(3).set('Visible',0)
                obj.handleAgent(2).set('Visible',1)
            end
            drawnow limitrate
        end
        
    end
    
end