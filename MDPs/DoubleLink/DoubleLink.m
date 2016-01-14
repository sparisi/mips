classdef DoubleLink < MDP
    
    %% Properties
    properties
        % Environment variables (state = [theta(1) thetad(1) theta(2) thetad(2)])
        lengths = [1 1];
        masses = [1 1];
        friction = [2.5, 2.5]';
        g = 9.81;
        dt = 0.002;
        
        % MDP variables
        dstate = 4;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds
        stateLB = [0, -50, 0, -50]';
        stateUB = [2*pi, 50, 2*pi, 50]';
        actionLB = [-20, -20]';
        actionUB = [20, 20]';
        rewardLB = -inf;
        rewardUB = 0;
    end
    
    properties
        target % function handle that defines the desired joint state at each timestep
        t % interal timestep (during an episode)
    end
    
    methods
        %% Constructor
        function obj = DoubleLink(target)
            obj.target = target;
        end
        
        %% Simulator
        function state = initstate(obj,n)
            state = repmat([pi 0 0 0]',1,n);
            obj.t = obj.dt;
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            % Check action
            real_action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
            penalty_action = matrixnorms(real_action-action,2);
            action = real_action;

            % State transition
            [gravity, coriolis, invM] = obj.getDynamicsMatrices(state);
            qdd = mtimes32(invM, action - coriolis - gravity);
            qd = state(2:2:end,:) + obj.dt * qdd;
            q = state(1:2:end,:) + obj.dt * qd;
            q = wrapin2pi(q);

            % Check velocity 
            real_vel = bsxfun(@max, bsxfun(@min,qd,obj.stateUB(2:2:end)), obj.stateLB(2:2:end));
            penalty_vel = matrixnorms(real_vel-qd,2);
            qd = real_vel;
            
            nextstate = [q(1,:); qd(1,:); q(2,:); qd(2,:)];

            % Compute reward
            qdes = obj.target.q(obj.t);
            qddes = obj.target.qd(obj.t);
            goalstate = [qdes(1), qddes(1), qdes(2), qddes(2)]';
            distance = bsxfun(@minus,nextstate,goalstate);
            reward = - matrixnorms(distance,2).^2;

            absorb = false(1,size(state,2)); % Infinite horizon
            obj.t = obj.t + obj.dt;

            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Dynamics
        function [gravity, coriolis, invM] = getDynamicsMatrices(obj, state)
            inertias = obj.masses .* (obj.lengths.^2 + 0.0001) ./ 12.0;
            m1 = obj.masses(1);
            m2 = obj.masses(2);
            l1 = obj.masses(1);
            l2 = obj.masses(2);
            lg1 = l1 / 2;
            lg2 = l2 / 2;
            q1 = state(1,:) + pi/2;
            q2 = state(3,:);
            q1d = state(2,:);
            q2d = state(4,:);
            s1 = sin(q1);
            c1 = cos(q1);
            s2 = sin(q2);
            c2 = cos(q2);
            s12 = sin(q1 + q2);
            c12 = cos(q1 + q2);
            M11 = m1 * lg1^2 + inertias(1) + m2 * (l1^2 + lg2^2 + 2 * l1 * lg2 * c2) + inertias(2);
            M12 = m2 * (lg2^2 +  l1 * lg2 * c2) + inertias(2);
            M21 = m2 * (lg2^2 +  l1 * lg2 * c2) + inertias(2);
            M22 = repmat(m2 * lg2^2 + inertias(2), 1, size(state,2));
            invdetM = 1 ./ (M11 .* M22 - M12 .* M21);
            invM(1,1,:) = bsxfun(@times, M22, invdetM);
            invM(1,2,:) = bsxfun(@times, -M21, invdetM);
            invM(2,1,:) = bsxfun(@times, M11, invdetM);
            invM(2,2,:) = bsxfun(@times, -M12, invdetM);
            gravity = [m1 * obj.g * lg1 * c1 + m2 * obj.g * (l1 * c1 + lg2 * c12);
                m2 * obj.g * lg2 * c12];
            coriolis = [2 * m2 * l1 * lg2 * (q1d .* q2d .* s2 + q1d.^2 .* s2) + obj.friction(1) * q1d;
                2 * m2 * l1 * lg2 * (q1d.^2 .* s2) + obj.friction(2) .* q2d];
        end

        %% Kinematics
        function X = getJointsInTaskSpace(obj, state)
            % Each column of X is a state in the task space (x1,y1,xd1,yd1,x2,y2,xd1,yd2)
            xy1 = obj.lengths(1) .* [sin(state(1,:)); cos(state(1,:))];
            xy2 = xy1 + obj.lengths(2) .* [sin(state(3,:)+state(1,:)); cos(state(3,:)+state(1,:))];
            
            xy1d = obj.lengths(1) .* ...
                [state(2,:).*cos(state(1,:)); ...
                -state(2,:).*sin(state(1,:))];
            xy2d = xy1d + obj.lengths(2) .* ...
                [(state(4,:)+state(2,:)).*cos(state(3,:)+state(1,:)); ...
                -(state(4,:)+state(2,:)).*sin(state(3,:)+state(1,:))];
            
            X = [xy1; xy1d; xy2; xy2d];
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            line(sum(obj.lengths)*[-1.1 1.1], [0 0], 'LineStyle', '--');
            axis(sum(obj.lengths)*[-1.1 1.1 -1.1 1.1]);

            lw = 4.0;
            colors = {[0.1 0.1 0.4], [0.4 0.4 0.8]};
            obj.handleAgent{1} = line([0 0], [0, 0], 'linewidth', lw, 'color', colors{1});
            obj.handleAgent{2} = line([0 0], [0, 0], 'linewidth', lw, 'color', colors{2});
            obj.handleAgent{3} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor',colors{1});
            obj.handleAgent{4} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor',colors{2});
            
            % 'Ghost' desired state
            obj.handleAgent{5} = line([0 0], [0, 0], 'linewidth', lw, 'color', 'r');
            obj.handleAgent{6} = line([0 0], [0, 0], 'linewidth', lw, 'color', 'r');
            obj.handleAgent{7} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor','r');
            obj.handleAgent{8} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor','r');
            obj.handleAgent{5}.Color(4) = 0.3;
            obj.handleAgent{6}.Color(4) = 0.3;
            obj.handleAgent{7}.FaceColor(4) = 0.1;
            obj.handleAgent{7}.EdgeColor(4) = 0.1;
            obj.handleAgent{8}.FaceColor(4) = 0.1;
            obj.handleAgent{8}.EdgeColor(4) = 0.1;
        end
        
        function updateplot(obj, state)
            r = 0.1;

            X = obj.getJointsInTaskSpace(state);
            xy1 = X(1:2,:);
            xy2 = X(5:6,:);
            
            obj.handleAgent{1}.XData = [0 xy1(1)];
            obj.handleAgent{1}.YData = [0 xy1(2)];
            obj.handleAgent{2}.XData = [xy1(1) xy2(1)];
            obj.handleAgent{2}.YData = [xy1(2) xy2(2)];
            obj.handleAgent{3}.Position = [xy1(1)-r,xy1(2)-r,2*r,2*r];
            obj.handleAgent{4}.Position = [xy2(1)-r,xy2(2)-r,2*r,2*r];

            % 'Ghost' desired state
            qdes = obj.target.q(obj.t);
            qddes = obj.target.qd(obj.t);
            goalstate = [qdes(1), qddes(1), qdes(2), qddes(2)]';
            X = obj.getJointsInTaskSpace(goalstate);
            xy1 = X(1:2,:);
            xy2 = X(5:6,:);
            obj.handleAgent{5}.XData = [0 xy1(1)];
            obj.handleAgent{5}.YData = [0 xy1(2)];
            obj.handleAgent{6}.XData = [xy1(1) xy2(1)];
            obj.handleAgent{6}.YData = [xy1(2) xy2(2)];
            obj.handleAgent{7}.Position = [xy1(1)-r,xy1(2)-r,2*r,2*r];
            obj.handleAgent{8}.Position = [xy2(1)-r,xy2(2)-r,2*r,2*r];
            drawnow limitrate
        end
        
    end
    
end