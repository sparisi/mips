classdef Reacher < MDP
% Planar two-link robot arm.
% The state is the joints angle and velocity, the action is the joints
% acceleration.
% The reward depends on the distance of the end-effector to the target + 
% a penalty on the velocity and the acceleration.
%
% =========================================================================
% REFERENCE
% V Tangkaratt, H van Hoof, S Parisi, M Sugiyama, G Neumann, J Peters
% Policy Search with High-Dimensional Context Variables (2017)
    
    properties
        % Environment variables
        l1 = 15/2; % Length of the first joint
        l2 = 15/2; % Length of the second joint
        dt = 0.1; 
        tar_x = 0; % Target in task space
        tar_y = 15;
        j1x = 0; % X coord of the base
        j1y = 0; % Y coord of the base
        
        % MDP variables
        dstate = 4;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;

        % Bounds : state = [q1 qd1 q2 qd2]
        stateLB = [-pi, -inf, -pi, -inf]';
        stateUB = [pi, inf, pi, inf]';
        actionLB = [-10, -10]';
        actionUB = [10, 10]';
        rewardLB = -15^2 - 0.001*10^2;
        rewardUB = 0;
    end
    
    methods
        
        %% Simulation
        function state = init(obj, n)
            if nargin == 1, n = 1; end
%             randpos = repmat([-pi/2; 0],1,n);
            randpos = myunifrnd(obj.stateLB(1:2:end),obj.stateUB(1:2:end),n);
            randvel = myunifrnd(-1*ones(1,obj.daction),1*ones(1,obj.daction),n);
            state = zeros(obj.dstate,n);
            state(1:2,:) = randpos;
            state(3:4,:) = randvel;
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            original_action = action;
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);

            q = state(1:2,:);
            qd = state(3:4,:);
            qdd = action;

            [eff_x, eff_y] = obj.kinematics(q);
            distance2 = sum(bsxfun(@minus,[eff_x;eff_y],[obj.tar_x;obj.tar_y]).^2,1);
            reward = - distance2 - 0.1*sum(qd.^2,1) - 0.01*sum(original_action.^2,1);
            
            [q, qd] = obj.dynamics(q, qd, qdd);
            q = wrapinpi(q);
            nextstate = [q; qd];
            absorb = false(size(reward));
        end
        
        %% Dynamics and kinematics
        function [q_next, qd_next] = dynamics(obj, q, qd, qdd)
            qd_next = qd + obj.dt * qdd;
            q_next = q + obj.dt * qd;
        end
        
        function [eff_x, eff_y, j2x, j2y] = kinematics(obj, q)
            j2x = obj.j1x + obj.l1*cos(q(1,:)); % Second joint x coord
            j2y = obj.j1y + obj.l1*sin(q(1,:)); % Second joint y coord
            eff_x = j2x + obj.l2*cos(q(2,:)+q(1,:)); % End-effector x coord
            eff_y = j2y + obj.l2*sin(q(2,:)+q(1,:)); % End-effector y coord
        end
        
    end

    %% Plotting
    methods(Hidden)
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            line_width = 5;
            marker_size = 10;
            
            plot(obj.j1x, obj.j1y, 'ok', 'markersize', marker_size, 'markerfaceColor','k'); % Joint1
            obj.handleAgent{1} = plot([0, 0], [0, 0], '-b', 'linewidth', line_width); % Line 1
            obj.handleAgent{2} = plot([0, 0], [0, 0],'-r', 'linewidth', line_width); % Line 2
            obj.handleAgent{3} = plot(0, 0, 'ob', 'markersize', marker_size, 'markerfaceColor','b'); % Joint2
            obj.handleAgent{4} = plot(0, 0, 'or', 'markersize', marker_size, 'markerfaceColor','r'); % Endeff
            
            xlim([-20, 20])
            ylim([-20, 20])
        end
        
        function updateplot(obj, state)
            [eff_x, eff_y, j2x, j2y] = obj.kinematics(state(1:2,:));
            obj.handleAgent{1}.XData = [obj.j1x, j2x];
            obj.handleAgent{1}.YData = [obj.j1y, j2y];
            obj.handleAgent{2}.XData = [j2x, eff_x];
            obj.handleAgent{2}.YData = [j2y, eff_y];
            obj.handleAgent{3}.XData = j2x;
            obj.handleAgent{3}.YData = j2y;
            obj.handleAgent{4}.XData = eff_x;
            obj.handleAgent{4}.YData = eff_y;
            
            drawnow limitrate
        end
    end
    
end