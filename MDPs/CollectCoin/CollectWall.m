classdef CollectWall < MDP
    
    %% Properties
    properties
        % Environment variables
        goal = [0.9; 0.8];
        coin = [0.2; 0.1];
        hole_y = 0.5;
        hole_width = 0.2;
        step = 0.05;
        step_hole = 0.01;
        radius_agent = 0.1;
                
        % MDP variables
        dstate = 4;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;
        
        % Bounds
        stateLB = zeros(4,1); % (x,y) of the agent + (x) of the center of the hole + boolean hasCoin
        stateUB = ones(4,1);
        actionLB = -ones(2,1);
        actionUB = ones(2,1)
        rewardLB = -2;
        rewardUB = 1;
    end
    
    methods
        
        %% Simulator
        function state = init(obj, n)
            agent = rand(2,n);
            if n == 1
                while agent(2) > obj.hole_y-obj.radius_agent && agent(2) < obj.hole_y+obj.radius_agent 
                    agent = rand(2,1); % Avoid init agent pos on the wall 
                end
            else
                n1 = ceil(n/2); 
                agent = [myunifrnd([0,0],[1,obj.hole_y-obj.radius_agent],n1), ... % Avoid init agent pos on the wall 
                    myunifrnd([0,obj.hole_y+obj.radius_agent],[1,1],n-n1)]; % (collect half on the left, half on the right)
            end
            hole_center = rand(1,n);
            has_coin = false(1,n);
            state = [agent; hole_center; has_coin];
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstates = size(state,2);

            % Init reward and absorb
            absorb = false(1, nstates);
            reward = zeros(1, nstates);
            
            % Parse state
            agent = state(1:2,:);
            hole_center = state(3,:);
            has_coin = state(4,:);

            % Parse action
            action = bsxfun(@times, action, 1 ./ max(matrixnorms(action,2),1));
            
            % Move agent
            next_agent = agent + action * obj.step;
            out_of_bounds = any(bsxfun(@le, next_agent, obj.stateLB(1:2)) | bsxfun(@ge, next_agent, obj.stateUB(1:2)));
            crossing = ( (agent(2,:) < obj.hole_y) & (next_agent(2,:) > obj.hole_y) ) ...
                | ( (agent(2,:) > obj.hole_y) & (next_agent(2,:) < obj.hole_y) );
            blocked = crossing & ~( agent(1,:) > hole_center - obj.hole_width/2 & agent(1,:) < hole_center + obj.hole_width/2 );
            next_agent = bsxfun(@max, bsxfun(@min,next_agent,obj.stateUB(1:2)), obj.stateLB(1:2)); % Put agent back in bounds
            next_agent(:,blocked) = agent(:,blocked);
            
            % Move barrier
            next_barrier = mod(hole_center + obj.step_hole, 1);

            % Reward
            dist_coin = abs( matrixnorms(bsxfun(@minus, agent, obj.coin), 2) );
            dist_goal = abs( matrixnorms(bsxfun(@minus, agent, obj.goal), 2) );
            to_deliver = has_coin & ~absorb;
            reward(to_deliver) = -dist_goal(to_deliver);
            to_pick = ~has_coin & ~absorb;
            reward(to_pick) = -dist_coin(to_pick);

            % Check coin delivered
            delivered = has_coin & obj.at_pos(next_agent, obj.goal);
            absorb(delivered) = true;
            reward(delivered) = 1;
            
            % Check coin picked
            next_has_coin = has_coin;
            next_has_coin(obj.at_pos(next_agent, obj.coin)) = true;
            reward(next_has_coin & ~has_coin) = 1;
            
            % Penalties
            reward(out_of_bounds | blocked) = reward(out_of_bounds | blocked) - 1;

            % Create next state
            nextstate = [next_agent
                next_barrier
                next_has_coin];

            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
        %% Helper
        function flag = at_pos(obj, pos_1, pos_2)
            flag = matrixnorms(bsxfun(@minus, pos_2, pos_1), 2) < obj.radius_agent;
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            plot([0,1],[obj.hole_y,obj.hole_y],'k-','LineWidth',2); % Barrier
            obj.handleAgent{3} = plot([-1 -1],[obj.hole_y obj.hole_y],'w-','LineWidth',2); % Hole
            plot(obj.goal(1),obj.goal(2),'ko','MarkerSize',12,'MarkerFaceColor','g'); % Goal
            obj.handleAgent{1} = plot(-1,-1,'ko','MarkerSize',12,'MarkerFaceColor','b'); % Agent
            obj.handleAgent{2} = plot(obj.coin(1),obj.coin(2),'ko','MarkerSize',12,'MarkerFaceColor','y'); % Coin
            
            axis([obj.stateLB(1), obj.stateUB(1), obj.stateLB(2), obj.stateUB(2)])
        end
        
        function updateplot(obj, state)
            agent = state(1:2,:);
            hole = state(3,:);
            has_coin = state(4,:);

            % Agent
            obj.handleAgent{1}.XData = agent(1);
            obj.handleAgent{1}.YData = agent(2);
            
            % Coin
            if has_coin, set(obj.handleAgent{2},'Visible','off'); 
            else set(obj.handleAgent{2},'Visible','on'); end
            
            % Hole
            obj.handleAgent{3}.XData = [hole(1)-obj.hole_width/2, hole(1)+obj.hole_width/2];

            drawnow limitrate
        end
        
        function [pixels, clims, cmap] = render(obj, state)
            steps_pixels = obj.step;
            tot_size = ceil(1/steps_pixels)+1;
            
            if nargin == 1, pixels = tot_size^2; return, end
            
            wall_value = -20;
            agent_value = -5;
            goal_value = 5;
            coin_value = 20;
            
            clims = [-20, 20];
            cmap = [0 0 0; 0 0 255; 0 180 180; 0 255 0; 255 255 0] / 255;
            
            agent = state(1:2,:);
            hole_center = state(3,:);
            has_coin = state(4,:);

            n = size(state,2);
            pixels = zeros(tot_size,tot_size,n);
            
            % Wall
            pixels(:,floor(obj.hole_y/steps_pixels)+1,:) = wall_value;
            pixels(:,ceil(obj.hole_y/steps_pixels)+1,:) = wall_value;
            
            % Goal
            goal_pixels = floor(obj.goal/steps_pixels) + 1;
            pixels(goal_pixels(1,:),goal_pixels(2,:),:) = goal_value;
            
            % Coin
            coin_pixels = floor(obj.coin/steps_pixels) + 1;
            pixels(coin_pixels(1,:),coin_pixels(2,:),has_coin==0) = coin_value;
            
            % Hole
            hole_init = ( ceil( (hole_center-obj.hole_width)/steps_pixels )+1);
            hole_end = ( floor( (hole_center+obj.hole_width)/steps_pixels )+1);
            hole_pixels_x = floor( linspaceNDim(hole_init, hole_end, max(hole_end-hole_init)+1) );
            hole_pixels_x = min(max(hole_pixels_x,1),tot_size);
            hole_pixels_x = hole_pixels_x(:);
            hole_pixels_y = repmat(floor(obj.hole_y/steps_pixels)+1,length(hole_pixels_x),1);
            hole_pixels_z = repmat((1:n)',length(hole_pixels_x)/n,1);
            pixels(sub2ind(size(pixels), hole_pixels_x, hole_pixels_y, hole_pixels_z)) = 0;

            % Remember wall (or agent will override it)
            old_pixels = pixels;
            
            % Agent cross
            agent_pixels = floor(agent/steps_pixels)+1;
            pixels(sub2ind(size(pixels), agent_pixels(1,:),agent_pixels(2,:), 1:n)) = agent_value;
            agent_pixels_right = min(bsxfun(@plus, agent_pixels, [0;1]), tot_size);
            pixels(sub2ind(size(pixels), agent_pixels_right(1,:),agent_pixels_right(2,:), 1:n)) = agent_value;
            agent_pixels_left = max(bsxfun(@plus, agent_pixels, [0;-1]), 1);
            pixels(sub2ind(size(pixels), agent_pixels_left(1,:),agent_pixels_left(2,:), 1:n)) = agent_value;
            agent_pixels_up = min(bsxfun(@plus, agent_pixels, [1;0]), tot_size);
            pixels(sub2ind(size(pixels), agent_pixels_up(1,:),agent_pixels_up(2,:), 1:n)) = agent_value;
            agent_pixels_down = max(bsxfun(@plus, agent_pixels, [-1;0]), 1);
            pixels(sub2ind(size(pixels), agent_pixels_down(1,:),agent_pixels_down(2,:), 1:n)) = agent_value;
            
            pixels(old_pixels == wall_value) = wall_value;
        end
        
    end
    
end