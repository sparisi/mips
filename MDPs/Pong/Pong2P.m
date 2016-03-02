classdef Pong2P < MDP
% Based on the implementation by David Buckingham.
% http://www.mathworks.com/matlabcentral/fileexchange/31177-dave-s-matlab-pong
    
    properties(Constant)
        MIN_BALL_SPEED = .8;
        MAX_BALL_SPEED = 3;
        BALL_ACCELERATION = 0.05;
        PADDLE_SPEED = 1.3;
        B_FACTOR = 1;
        P_FACTOR = 2;
        Y_FACTOR = 0.01;
        
        BALL_RADIUS = 1.5;
        WALL_WIDTH = 3;
        FIGURE_WIDTH = 800;
        FIGURE_HEIGHT = 480;
        PLOT_W = 150;
        PLOT_H = 100;
        GOAL_SIZE = Pong2P.PLOT_H;
        GOAL_TOP = (Pong2P.PLOT_H + Pong2P.GOAL_SIZE) / 2;
        GOAL_BOT = (Pong2P.PLOT_H - Pong2P.GOAL_SIZE) / 2;
        PADDLE_H = 14;
        PADDLE_W = 2;
        OFFSET = 10;
        PADDLE_X1 = Pong2P.OFFSET;
        PADDLE_X2 = Pong2P.PLOT_W - Pong2P.OFFSET;
        
        FIGURE_COLOR = [0, 0, 0];
%         AXIS_COLOR = [.15, .15, .15];
        AXIS_COLOR = [0 0 0];
        CENTER_RADIUS = 15;
        BALL_MARKER_SIZE = 10;
%         BALL_COLOR = [.1, .7, .1];
%         BALL_OUTLINE = [.7, 1, .7];
        BALL_COLOR = [1 1 1];
        BALL_OUTLINE = [1 1 1];
        BALL_SHAPE = 'o';
        PADDLE_LINE_WIDTH = 2;
%         WALL_COLOR = [.3, .3, .8];
%         PADDLE_COLOR = [1, .5, 0];
        PADDLE_COLOR = [1 1 1];
        WALL_COLOR = [1 1 1];
        CENTERLINE_COLOR = Pong2P.PADDLE_COLOR .* .8;
    end
        
    properties
        % MDP variables
        dstate = 7;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.999;

        % Bounds : state = [ballX, ballY, ballSpeed, ballVectorX, ballVectorY, paddle1Y, paddle2Y])
        stateLB = [0
            0
            Pong2P.MIN_BALL_SPEED % Speed of the ball, it increases when a player hits it
            0 % ballVector has norm 1 and represents the ball speed direction
            0
            0
            0];
        stateUB = [Pong2P.PLOT_W - Pong2P.BALL_RADIUS
            Pong2P.PLOT_H - Pong2P.BALL_RADIUS
            Pong2P.MAX_BALL_SPEED
            1
            1
            Pong2P.PLOT_H - Pong2P.PADDLE_H
            Pong2P.PLOT_H - Pong2P.PADDLE_H];
        actionLB = 1;
        actionUB = 4;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods

        function state = initstate(obj, n)
            % Random init ball direction
            ballVector = [1-(2*rand(1,n)); 1-(2*rand(1,n))];
            ballVector(1,:) = ballVector(1,:) .* ((rand(1,n) / obj.B_FACTOR) + 1);
            ballVector = bsxfun(@times, ballVector, 1 ./ (sqrt(ballVector(1,:).^2 + ballVector(2,:).^2)));

            state = [
                obj.PLOT_W / 2 * ones(1, n) % ballX
                obj.PLOT_H / 2 * ones(1, n) % ballY
                obj.MIN_BALL_SPEED * ones(1, n) % ballSpeed
                ballVector
                obj.PLOT_H / 2 * ones(1, n) % paddle1Y
                obj.PLOT_H / 2 * ones(1, n) % paddle2Y
                ];

            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            n = size(state,2);
            
            % Parse state
            ballX = state(1,:);
            ballY = state(2,:);
            ballSpeed = state(3,:);
            ballVector = state(4:5,:);
            paddle1Y = state(6,:);
            paddle2Y = state(7,:);

            % Parse action
            move = [-1  1 -1 1
                    -1 -1  1 1];
            paddle1V = move(1,action);
            paddle2V = move(2,action);

            % Move paddles
            paddle1Y = paddle1Y + obj.PADDLE_SPEED * paddle1V;
            paddle2Y = paddle2Y + obj.PADDLE_SPEED * paddle2V;

            % Check environment bounds
            paddle1Y = min(paddle1Y, obj.PLOT_H - obj.PADDLE_H);
            paddle1Y = max(paddle1Y, obj.PADDLE_H);
            paddle2Y = min(paddle2Y, obj.PLOT_H - obj.PADDLE_H);
            paddle2Y = max(paddle2Y, obj.PADDLE_H);
            
            % Move ball
            newX = ballX + (ballSpeed .* ballVector(1,:));
            newY = ballY + (ballSpeed .* ballVector(2,:));

            % Check bounce
            tmpV = nan(2,n);
            idx = newY > (obj.PLOT_H - obj.BALL_RADIUS); % Hit top wall
            tmpV(:,idx) = [ballVector(1,idx); -1 * (obj.Y_FACTOR + abs(ballVector(2,idx)))];
            idx = newY < obj.BALL_RADIUS; % Hit bottom wall
            tmpV(:,idx) = [ballVector(1,idx); (obj.Y_FACTOR + abs(ballVector(2,idx)))];
            idx = newX < obj.PADDLE_X1 + obj.PADDLE_W + obj.BALL_RADIUS ... % Hit paddle1 right
                & newX > obj.PADDLE_X1 - obj.PADDLE_W - obj.BALL_RADIUS ... % Hit paddle1 left
                & newY < paddle1Y + obj.PADDLE_H + obj.BALL_RADIUS ... % Hit paddle1 top
                & newY > paddle1Y - obj.PADDLE_H - obj.BALL_RADIUS; % Hit paddle1 bottom
            tmpV(:,idx) = [(ballX(:,idx) - obj.PADDLE_X1) * obj.P_FACTOR; newY(:,idx) - paddle1Y(:,idx)];

            idx = newX < obj.PADDLE_X2 + obj.PADDLE_W + obj.BALL_RADIUS ... % Hit paddle2 right
                & newX > obj.PADDLE_X2 - obj.PADDLE_W - obj.BALL_RADIUS ... % Hit paddle2 left
                & newY < paddle2Y + obj.PADDLE_H + obj.BALL_RADIUS ... % Hit paddle2 top
                & newY > paddle2Y - obj.PADDLE_H - obj.BALL_RADIUS; % Hit paddle2 bottom
            tmpV(:,idx) = [(ballX(:,idx) - obj.PADDLE_X2) * obj.P_FACTOR; newY(:,idx) - paddle2Y(:,idx)];

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
                paddle1Y
                paddle2Y];

            % Reward and terminal condition
            absorb = false(1,n);
            reward = zeros(1,n);
            idx = newX < -obj.BALL_RADIUS | newX > obj.PLOT_W - obj.BALL_RADIUS;
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

            obj.handleAgent{3} = rectangle('Position', ...
                [obj.PADDLE_X2 - obj.PADDLE_W, obj.PLOT_H / 2 - obj.PADDLE_H, 2 * obj.PADDLE_W, 2 * obj.PADDLE_H], ...
                'FaceColor', obj.PADDLE_COLOR, ...
                'EdgeColor', obj.PADDLE_COLOR, ...
                'LineWidth', obj.PADDLE_LINE_WIDTH);
        end

        function obj = updateplot(obj, state)
            ballX = state(1,:);
            ballY = state(2,:);
            paddle1Y = state(6,:);
            paddle2Y = state(7,:);
            
            set(obj.handleAgent{1}, 'XData', ballX, 'YData', ballY);
            set(obj.handleAgent{2}, 'Position', ...
                [obj.PADDLE_X1 - obj.PADDLE_W, paddle1Y - obj.PADDLE_H, 2 * obj.PADDLE_W, 2 * obj.PADDLE_H]);
            set(obj.handleAgent{3}, 'Position', ...
                [obj.PADDLE_X2 - obj.PADDLE_W, paddle2Y - obj.PADDLE_H, 2 * obj.PADDLE_W, 2 * obj.PADDLE_H]);

            drawnow limitrate
        end

    end
    
end
