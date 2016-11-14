classdef PongEnv < MDP
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
        GOAL_SIZE = PongEnv.PLOT_H;
        GOAL_TOP = (PongEnv.PLOT_H + PongEnv.GOAL_SIZE) / 2;
        GOAL_BOT = (PongEnv.PLOT_H - PongEnv.GOAL_SIZE) / 2;
        PADDLE_H = 14;
        PADDLE_W = 2;
        OFFSET = 10;
        PADDLE_X1 = PongEnv.OFFSET;
        PADDLE_X2 = PongEnv.PLOT_W - PongEnv.OFFSET;
        
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
        CENTERLINE_COLOR = PongEnv.PADDLE_COLOR .* .8;
    end

end
