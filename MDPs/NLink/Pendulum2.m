classdef Pendulum2 < NLinkEnv
% REFERENCE
% M Deisenroth, C E Rasmussen, J Peters
% Gaussian Process Dynamic Programming (2009)

    %% Properties
    properties
        % Environment variables
        lengths = 1;
        masses = 1;
        g = 9.81;
        dt = 0.05;
        mu = 0.05; % Friction coeff
        
        q_des = 0; % Upright position
        
        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds : state = [q qd]
        stateLB = [-pi, -7]';
        stateUB = [pi, 7]';
        actionLB = -5;
        actionUB = 5;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods

        %% Simulation
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = repmat([-pi 0]',1,n);
            state = myunifrnd([-pi, -1], [pi, 1], n);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            q = state(1,:);
            qd = state(2,:);
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
            reward = - 1 + exp(-angdiff(q,obj.q_des,'rad').^2 - 0.2*qd.^2);
            qdd = (-obj.mu*qd + obj.masses*obj.g*obj.lengths*sin(q) + action) ...
                / (obj.masses*obj.lengths^2);
            qd = qd + qdd*obj.dt;
            q = q + qd*obj.dt;
            q = wrapinpi(q);
            qd = bsxfun(@max, bsxfun(@min,qd,obj.stateUB(2)), obj.stateLB(2));
            nextstate = [q; qd];
            absorb = false(1,size(state,2));
            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Kinematics
        function X = getJointsInTaskSpace(obj, state)
        % X = [ x y xd yd ]
            q = state(1,:);
            qd = state(2,:);
            xy = obj.lengths(1) .* [cos(q); sin(q)];
            xy1 = obj.lengths(1) .* [qd.*cos(q); -qd.*sin(q)];
            X = [xy; xy1];
        end

        %% Plotting
        function initplot(obj)
            initplot@NLinkEnv(obj)
            text(1,0.15,'0 (2\pi)','HorizontalAlignment','center')
            text(-1,0.15,'\pi (-\pi)','HorizontalAlignment','center')
            text(0,1,'\pi/2','HorizontalAlignment','center')
            text(0,-1,'-\pi/2','HorizontalAlignment','center')
            pbaspect([1 1 1])
%             rectangle('Position',[-1,-1,2,2],'Curvature',[1,1]);
        end
    end
     
end