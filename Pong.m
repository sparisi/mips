classdef Pong < MDP
    
    %% Properties
    properties
        % Environment variables (see reference for details)
        FRAME_DELAY = 0.01;
        MIN_BALL_SPEED = 0.8;
        MAX_BALL_SPEED = 3;
        BALL_ACCELERATION = 0.05;
        PADDLE_SPEED = 1.3;
        B_FACTOR = 1;
        P_FACTOR = 2;
        Y_FACTOR = 0.01;
        GOAL_BUFFER = 5;
        BALL_RADIUS = 1.5;
        WALL_WIDTH = 3;
        FIGURE_WIDTH = 800;
        FIGURE_HEIGHT = 480;
        PLOT_W = 150;
        PLOT_H = 100;
        GOAL_SIZE = 50;
        GOAL_TOP = (PLOT_H + GOAL_SIZE) / 2;
        GOAL_BOT = (PLOT_H - GOAL_SIZE) / 2;
        PADDLE_H = 18;
        PADDLE_W = 3;
        PADDLE = [0 PADDLE_W PADDLE_W 0 0; PADDLE_H PADDLE_H 0 0 PADDLE_H];
        PADDLE_SPACE = 10;
        FIGURE_COLOR = [0, 0, 0];
        AXIS_COLOR = [.15, .15, .15];
        CENTER_RADIUS = 15;
        BALL_MARKER_SIZE = 10;
        BALL_COLOR = [.1, .7, .1];
        BALL_OUTLINE = [.7, 1, .7];
        BALL_SHAPE = 'o';
        PADDLE_LINE_WIDTH = 2;
        WALL_COLOR = [.3, .3, .8];
        PADDLE_COLOR = [1, .5, 0];
        CENTERLINE_COLOR = PADDLE_COLOR .* .8;
        PAUSE_BACKGROUND_COLOR = FIGURE_COLOR;
        PAUSE_TEXT_COLOR = [.9, .9, .9];
        PAUSE_EDGE_COLOR = BALL_COLOR;
        TITLE_COLOR = 'w';


fig = []; %main program figure
score = []; %1x2 vector holding player scores
winner = []; %normally 0. 1 if player1 wins, 2 if player2 wins
ballPlot = []; %main plot, includes ball and walls
paddle1Plot = []; %plot for paddle
paddle2Plot = [];
ballVector=[]; %normalized vector for ball movement
ballSpeed=[];
ballX = []; %ball location
ballY = [];
paddle1V = []; %holds either 0, -1, or 1 for paddle movement
paddle2V = [];
paddle1 = []; %2x5 matrix describing paddle, based on PADDLE
paddle2 = [];        
        
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.9;

        % Bounds : state = [ballX, ballY, ballSpeed, paddle1, paddle2, score1, score2])
        stateLB = [-3, -inf, -pi, -inf]';
        stateUB = [3, inf, pi, inf]';
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj,n)
            state = [
                obj.PLOT_W / 2 % ballX
                obj.PLOT_H / 2; % ballY
                xxx % ballXd
                xxx % ballYd
                xxx % paddle1
                0 % paddle1d
                xxx % paddle2
                0 % paddle2d
                0 % score1
                0 % score2
                ];
            
            PADDLE = [0 PADDLE_W PADDLE_W 0 0; PADDLE_H PADDLE_H 0 0 PADDLE_H];
            
            
            paddle1 = [PADDLE(1,:)+PADDLE_SPACE; ...
                PADDLE(2,:)+((PLOT_H - PADDLE_H)/2)];
            paddle2 = [PADDLE(1,:)+ PLOT_W - PADDLE_SPACE - PADDLE_W; ...
                PADDLE(2,:)+((PLOT_H - PADDLE_H)/2)];
            
            resetGame;
            
            bounce([1-(2*rand), 1-(2*rand)]);
            ballSpeed = MIN_BALL_SPEED;
        end
        
        
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            obj.moveBall();
            obj.movePaddle();



        end
        
        
        %% 
        function [ballSpeed, ballVector] = bounce(ballSpeed, tempV)
            % Increase ballXd by a random amount. It helps keeping the ball 
            % moving more horizontally than vertically.
            tempV(1) = tempV(1) * ((rand/obj.B_FACTOR) + 1);
            tempV = tempV ./ (sqrt(tempV(1)^2 + tempV(2)^2)); % normalize vector
            ballVector = tempV;
            if (ballSpeed + obj.BALL_ACCELERATION < obj.MAX_BALL_SPEED) % bouncing accelerates ball
                ballSpeed = ballSpeed + obj.BALL_ACCELERATION;
            end
        end


        function moveBall(obj, ballX, ballY, ballSpeed, ballVector)
            %paddle boundaries, useful for hit testing ball
            p1T = paddle1(2,1);
            p1B = paddle1(2,3);
            p1L = paddle1(1,1);
            p1R = paddle1(1,2);
            p1Center = ([p1L p1B] + [p1R p1T]) ./ 2;
            p2T = paddle2(2,1);
            p2B = paddle2(2,3);
            p2L = paddle2(1,1);
            p2R = paddle2(1,2);
            p2Center = ([p2L p2B] + [p2R p2T]) ./ 2;
            
            %while hit %calculate new vectors until we know it wont hit something
            %temporary new ball location, only apply if ball doesn't hit anything.
            newX = ballX + (ballSpeed * ballVector(1));
            newY = ballY + (ballSpeed * ballVector(2));
            
            %hit test right wall
            if ( newX > (obj.PLOT_W - obj.BALL_RADIUS) && ...
                    (ballY < obj.GOAL_BOT + obj.BALL_RADIUS ...
                    || newY > obj.GOAL_TOP - obj.BALL_RADIUS) )
                if (newY > obj.GOAL_BOT && newY < obj.GOAL_TOP - obj.BALL_RADIUS) % hit right wall
                    [ballSpeed, ballVector] = bounce([newX - obj.PLOT_W, newY - obj.GOAL_BOT]); % hit bottom goal edge
                elseif (newY < GOAL_TOP && newY > GOAL_BOT + BALL_RADIUS)
                    [ballSpeed, ballVector] = bounce([newX - PLOT_W, newY - GOAL_TOP]); % hit top goal edge
                else
                    %hit flat part of right wall
                    [ballSpeed, ballVector] = bounce([-1 * abs(ballVector(1)), ballVector(2)]);
                end
                
                %hit test left wall
            elseif (newX < BALL_RADIUS ...
                    && (newY<GOAL_BOT+BALL_RADIUS || newY>GOAL_TOP-BALL_RADIUS))
                %hit left wall
                if (newY > GOAL_BOT && newY < GOAL_TOP - BALL_RADIUS)
                    %hit bottom goal edge
                    [ballSpeed, ballVector] = bounce([newX, newY - GOAL_BOT]);
                elseif (newY < GOAL_TOP && newY > GOAL_BOT + BALL_RADIUS)
                    %hit top goal edge
                    [ballSpeed, ballVector] = bounce([newX, newY - GOAL_TOP]);
                else
                    [ballSpeed, ballVector] = bounce([abs(ballVector(1)), ballVector(2)]);
                end
                
                %hit test top wall
            elseif (newY > (PLOT_H - BALL_RADIUS))
                %hit top wall
                [ballSpeed, ballVector] = bounce([ballVector(1), -1 * (Y_FACTOR + abs(ballVector(2)))]);
                %hit test bottom wall
            elseif (newY < BALL_RADIUS)
                %hit bottom wall,
                [ballSpeed, ballVector] = bounce([ballVector(1), (Y_FACTOR + abs(ballVector(2)))]);
                
                %hit test paddle 1
            elseif (newX < p1R + BALL_RADIUS ...
                    && newX > p1L - BALL_RADIUS ...
                    && newY < p1T + BALL_RADIUS ...
                    && newY > p1B - BALL_RADIUS)
                [ballSpeed, ballVector] = bounce([(ballX-p1Center(1)) * P_FACTOR, newY-p1Center(2)]);
                
                %hit test paddle 2
            elseif (newX < p2R + BALL_RADIUS ...
                    && newX > p2L - BALL_RADIUS ...
                    && newY < p2T + BALL_RADIUS ...
                    && newY > p2B - BALL_RADIUS)
                [ballSpeed, ballVector] = bounce([(ballX-p2Center(1)) * P_FACTOR, newY-p2Center(2)]);
            else
                % No hits
            end
            
            %move ball to new location
            ballX = newX;
            ballY = newY;
            
        end
        
%------------movePaddles------------
%uses paddle velocity set paddles
%called from main loop on every frame
  function movePaddles
    %set new paddle y locations
    paddle1(2,:) = paddle1(2,:) + (PADDLE_SPEED * paddle1V);
    paddle2(2,:) = paddle2(2,:) + (PADDLE_SPEED * paddle2V);
    %if paddle out of bounds, move it in bounds
    if paddle1(2,1) > PLOT_H
      paddle1(2,:) = PADDLE(2,:) + PLOT_H - PADDLE_H;
    elseif paddle1(2,3) < 0
      paddle1(2,:) = PADDLE(2,:);
    end
    if paddle2(2,1) > PLOT_H
      paddle2(2,:) = PADDLE(2,:) + PLOT_H - PADDLE_H;
    elseif paddle2(2,3) < 0
      paddle2(2,:) = PADDLE(2,:);
    end
  end


    function checkGoal
    goal = false;
    
    if ballX > PLOT_W + BALL_RADIUS + GOAL_BUFFER
      score(1) = score(1) + 1;
      if score(1) == MAX_POINTS;
        winner = 1;
      end
      goal = true;
    elseif ballX < 0 - BALL_RADIUS - GOAL_BUFFER
      score(2) = score(2) + 1;
      if score(2) == MAX_POINTS;
        winner = 2;
      end
      goal = true;
    end






    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            obj.handleAgent{1} = plot(0,0,'k','LineWidth',6); % Cart
            obj.handleAgent{2} = plot(0,0,'ko','MarkerSize',12,'MarkerEdgeColor','k','MarkerFaceColor','k'); % Cart-Pole Link / Wheel
            obj.handleAgent{3} = plot(0,0,'k','LineWidth',4); % Pole
            
            plot([obj.stateLB(1),obj.stateLB(1)],[-10,10],'r','LineWidth',2)
            plot([obj.stateUB(1),obj.stateUB(1)],[-10,10],'r','LineWidth',2)

            axis([-3 3 -1.5 1.5]);
        end
        
        function updateplot(obj, state)
    set(ballPlot, 'XData', ballX, 'YData', ballY);
    set(paddle1Plot, 'Xdata', paddle1(1,:), 'YData', paddle1(2,:));
    set(paddle2Plot, 'Xdata', paddle2(1,:), 'YData', paddle2(2,:));
    drawnow;
    pause(FRAME_DELAY);
        end
        
    end
    
end