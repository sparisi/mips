classdef GridworldSparseWall < GridworldEnv
% Like GridworldSparse, but with a wall which makes it harder to reach the
% highest reward.
    
    %% Properties
    properties
        % Environment variables
        reward = zeros(50,50);
        isopen = ones(50,50);
        
        % Finite states and actions
        allstates = allcomb(1:50, 1:50);
        
        % Bounds
        stateLB = [1 1]';
        stateUB = [50 50]';
        rewardLB
        rewardUB
    end
    
    methods
        
        %% Constructor
        function obj = GridworldSparseWall()
            obj.reward(50,1) = 10000;
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
            if nargin == 1, n = 1; end
            state = 25 * ones(2,n);
%             state = [randi(obj.stateUB(1),1,n); randi(obj.stateUB(2),1,n)];
        end
        
    end

    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            cells = obj.reward;
            wall_value = -10000;
            cells(~obj.isopen) = wall_value;
            h = image(flipud(cells)); % Plot environment

            imggrid(h,'k',0.5); % Add grid

            cells = flipud(cells)';
            [rows,cols] = find(cells);
            for i = 1 : length(rows) % Add value
                if ~ (cells(rows(i),cols(i)) == wall_value)
                    if cells(rows(i),cols(i)) > 0
                        c = 'green';
                    else
                        c = 'red';
                    end
                    text('position', [rows(i) cols(i)], ...
                        'fontsize', 10, ...
                        'string', num2str(cells(rows(i),cols(i))), ...
                        'color', c, ...
                        'horizontalalignment', 'center')
                end
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
        
    end
    
end