classdef Pong1P < PongEnv
% 1-player pong (the wall replaces the second player).
% The goal is to play as long as possible.

    properties
        % MDP variables
        dstate = 6;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.999;

        % Finite actions
        allactions = [-1  1];

        % Bounds : state = [ballX, ballY, ballSpeed, ballVectorX, ballVectorY, paddleY])
        stateLB = [0
            0
            PongEnv.MIN_BALL_SPEED % Speed of the ball, it increases when the player hits it
            0 % ballVector has norm 1 and represents the ball speed direction
            0
            0];
        stateUB = [PongEnv.PLOT_W - PongEnv.BALL_RADIUS
            PongEnv.PLOT_H - PongEnv.BALL_RADIUS
            PongEnv.MAX_BALL_SPEED
            1
            1
            PongEnv.PLOT_H - PongEnv.PADDLE_H];
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods

        function state = init(obj, n)
            % Random init ball direction
            ballVector = [1-(2*rand(1,n)); 1-(2*rand(1,n))];
            ballVector(1,:) = ballVector(1,:) .* ((rand(1,n) / obj.B_FACTOR) + 1);
            ballVector = bsxfun(@times, ballVector, 1 ./ (sqrt(ballVector(1,:).^2 + ballVector(2,:).^2)));

            state = [
                obj.PLOT_W / 2 * ones(1, n) % ballX
                obj.PLOT_H / 2 * ones(1, n) % ballY
                obj.MIN_BALL_SPEED * ones(1, n) % ballSpeed
                ballVector
                obj.PLOT_H / 2 * ones(1, n) % paddleY
                ];
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            n = size(state,2);
            
            % Parse state
            ballX = state(1,:);
            ballY = state(2,:);
            ballSpeed = state(3,:);
            ballVector = state(4:5,:);
            paddleY = state(6,:);

            % Parse action
            paddleV = obj.allactions(action);

            % Move paddle
            paddleY = paddleY + obj.PADDLE_SPEED * paddleV;

            % Check environment bounds
            paddleY = min(paddleY, obj.PLOT_H - obj.PADDLE_H);
            paddleY = max(paddleY, obj.PADDLE_H);
            
            % Move ball
            newX = ballX + (ballSpeed .* ballVector(1,:));
            newY = ballY + (ballSpeed .* ballVector(2,:));

            % Check bounce
            tmpV = nan(2,n);
            idx = newX > (obj.PLOT_W - obj.BALL_RADIUS); % Hit right wall
            tmpV(:,idx) = [-1 * abs(ballVector(1,idx)); ballVector(2,idx)];
            idx = newY > (obj.PLOT_H - obj.BALL_RADIUS); % Hit top wall
            tmpV(:,idx) = [ballVector(1,idx); -1 * (obj.Y_FACTOR + abs(ballVector(2,idx)))];
            idx = newY < obj.BALL_RADIUS; % Hit bottom wall
            tmpV(:,idx) = [ballVector(1,idx); (obj.Y_FACTOR + abs(ballVector(2,idx)))];
            idx = newX < obj.PADDLE_X1 + obj.PADDLE_W + obj.BALL_RADIUS ... % Hit paddle right
                & newX > obj.PADDLE_X1 - obj.PADDLE_W - obj.BALL_RADIUS ... % Hit paddle left
                & newY < paddleY + obj.PADDLE_H + obj.BALL_RADIUS ... % Hit paddle top
                & newY > paddleY - obj.PADDLE_H - obj.BALL_RADIUS; % Hit paddle bottom
            tmpV(:,idx) = [(ballX(:,idx) - obj.PADDLE_X1) * obj.P_FACTOR; newY(:,idx) - paddleY(:,idx)];

            % Update speed and direction if ball bounced
            idx = ~isnan(tmpV(1,:));
            tmpV(1,idx) = tmpV(1,idx) .* ((rand(1,sum(idx)) / obj.B_FACTOR) + 1);
            tmpV(:,idx) = bsxfun(@times, tmpV(:,idx), 1 ./ (sqrt(tmpV(1,idx).^2 + tmpV(2,idx).^2)));
            ballVector(:,idx) = tmpV(:,idx);
            ballSpeed(:,idx) = min(ballSpeed(:,idx) + obj.BALL_ACCELERATION, obj.MAX_BALL_SPEED);
            
            % Update state
            nextstate = [newX
                newY
                ballSpeed
                ballVector
                paddleY];

            % Reward and terminal condition
            absorb = false(1,n);
            reward = zeros(1,n);
            idx = newX < -obj.BALL_RADIUS;
            reward(idx) = -1;
            absorb(idx) = true;
            
            if obj.realtimeplot, obj.updateplot(nextstate); end
        end

    end


    %% Plotting
    methods(Hidden = true)

        function obj = initplot(obj)
            scrsz = get(0,'ScreenSize');
            obj.handleEnv = figure('Position', [(scrsz(3) - obj.FIGURE_WIDTH) / 2 ...
                (scrsz(4) - obj.FIGURE_HEIGHT) / 2 ...
                obj.FIGURE_WIDTH, obj.FIGURE_HEIGHT]);
            set(obj.handleEnv, 'Resize', 'off');
            axis([0 obj.PLOT_W 0 obj.PLOT_H])
            axis manual
            set(gca, 'color', obj.AXIS_COLOR, 'YTick', [], 'XTick', []);
            set(obj.handleEnv, 'color', obj.FIGURE_COLOR);
            hold on
            
            topWallXs = [0 0 obj.PLOT_W obj.PLOT_W];
            topWallYs = [obj.GOAL_TOP obj.PLOT_H obj.PLOT_H obj.GOAL_TOP];
            bottomWallXs = [0 0 obj.PLOT_W obj.PLOT_W];
            bottomWallYs = [obj.GOAL_BOT 0 0 obj.GOAL_BOT];
            plot(topWallXs, topWallYs, '-', ...
                'LineWidth', obj.WALL_WIDTH, 'Color', obj.WALL_COLOR);
            plot(bottomWallXs, bottomWallYs, '-', ...
                'LineWidth', obj.WALL_WIDTH, 'Color', obj.WALL_COLOR);
            plot([obj.PLOT_W, obj.PLOT_W], [0, obj.PLOT_H], ...
                'LineWidth', obj.WALL_WIDTH, 'Color', obj.WALL_COLOR);

            centerline = plot([obj.PLOT_W/2, obj.PLOT_W/2], [obj.PLOT_H, 0], '--');
            set(centerline, 'Color', obj.CENTERLINE_COLOR);

            obj.handleAgent{1} = plot(obj.PLOT_W / 2, obj.PLOT_H / 2);
            set(obj.handleAgent{1}, 'Marker', obj.BALL_SHAPE);
            set(obj.handleAgent{1}, 'MarkerEdgeColor', obj.BALL_OUTLINE);
            set(obj.handleAgent{1}, 'MarkerFaceColor', obj.BALL_COLOR);
            set(obj.handleAgent{1}, 'MarkerSize', obj.BALL_MARKER_SIZE);
            
            obj.handleAgent{2} = rectangle('Position', ...
                [obj.PADDLE_X1 - obj.PADDLE_W, obj.PLOT_H / 2 - obj.PADDLE_H, 2 * obj.PADDLE_W, 2 * obj.PADDLE_H], ...
                'FaceColor', obj.PADDLE_COLOR, ...
                'EdgeColor', obj.PADDLE_COLOR, ...
                'LineWidth', obj.PADDLE_LINE_WIDTH);
        end

        function obj = updateplot(obj, state)
            ballX = state(1,:);
            ballY = state(2,:);
            paddleY = state(6,:);
            
            set(obj.handleAgent{1}, 'XData', ballX, 'YData', ballY);
            set(obj.handleAgent{2}, 'Position', ...
                [obj.PADDLE_X1 - obj.PADDLE_W, paddleY - obj.PADDLE_H, 2 * obj.PADDLE_W, 2 * obj.PADDLE_H]);

            drawnow limitrate
        end

    end
    
end
