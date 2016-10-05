classdef DoubleLink < MDP
% REFERENCE
% T Yoshikawa
% Foundations of Robotics: Analysis and Control (1990)

    %% Properties
    properties
        % Environment variables
        lengths = [1 1];
        masses = [1 1];
        inertias = [1 1] .* ([1 1].^2 + 0.0001) ./ 3.0;
        friction_coeff = [2.5 2.5]'; % Viscous friction coefficients
%         g = 9.81; % If 0, the problem becames a simpler planar reaching task
        g = 0;
        dt = 0.02;
        endEff_des = [0.5 -0.5]'; % Goal in task space
        q_des = [pi/2 0]'; % Goal in joint space
%         q_des = [pi/3 -pi/3]';
%         q_des = [-pi/4 0]';
        qd_des  = [0 0]';
        mode = 'joint';
%         mode = 'task';
        
        % MDP variables
        dstate = 4;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds : state = [q1 qd1 q2 qd2]
        stateLB = [-pi, -50, -pi, -50]';
        stateUB = [pi, 50, pi, 50]';
        actionLB = [-10, -10]';
        actionUB = [10, 10]';
        rewardLB = -inf;
        rewardUB = 0;
    end
    
    methods
        %% Simulator
        function state = initstate(obj,n)
            state = repmat([3/2*pi 0 0 0]',1,n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            % Constrain action
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);

            % State transition
            [gravity, coriolis, invM, friction] = obj.getDynamicsMatrices(state);
            qdd = mtimes32(invM, action - coriolis - gravity - friction);
            qd = state(2:2:end,:) + obj.dt * qdd;
            q = state(1:2:end,:) + obj.dt * qd;
            
            nextstate = [q(1,:); qd(1,:); q(2,:); qd(2,:)];

            % Compute reward
            switch obj.mode
                case 'joint'
                    distance = abs(angdiff(q,obj.q_des,'rad'));
                    reward = -matrixnorms(distance,2).^2 ...
                        - 0.05 * sum(abs(action),1); % Penalty on the action
                case 'task'
                    X = obj.getJointsInTaskSpace(nextstate);
                    endEffector = X(5:6,:);
                    distance2 = sum(bsxfun(@minus,endEffector,obj.endEff_des).^2,1);
                    reward = exp( -distance2 / (2*(1^2)) ) ... % The closer to the goal, the higher the reward
                        - 0.05 * sum(abs(action),1); % Penalty on the action
                otherwise
                    error('Unknown mode.')
            end

            absorb = false(1,size(state,2)); % Infinite horizon

            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Dynamics
        function [gravity, coriolis, invM, friction] = getDynamicsMatrices(obj, state)
            inertia1 = obj.inertias(1);
            inertia2 = obj.inertias(1);
            m1 = obj.masses(1);
            m2 = obj.masses(2);
            l1 = obj.lengths(1);
            l2 = obj.lengths(2);
            lg1 = l1 / 2;
            lg2 = l2 / 2;
            q1 = state(1,:);
            q2 = state(3,:);
            q1d = state(2,:);
            q2d = state(4,:);
%             s1 = sin(q1);
            c1 = cos(q1);
            s2 = sin(q2);
            c2 = cos(q2);
%             s12 = sin(q1 + q2);
            c12 = cos(q1 + q2);
            
            M11 = m1 * lg1^2 + inertia1 + m2 * (l1^2 + lg2^2 + 2 * l1 * lg2 * c2) + inertia2;
            M12 = m2 * (lg2^2 + l1 * lg2 * c2) + inertia2;
            M21 = M12;
            M22 = repmat(m2 * lg2^2 + inertia2, 1, size(state,2));
            invdetM = 1 ./ (M11 .* M22 - M12 .* M21);
            invM(1,1,:) = M22;
            invM(1,2,:) = -M21;
            invM(2,1,:) = -M12;
            invM(2,2,:) = M11;
            invM = bsxfun(@times, invM, permute(invdetM, [3 1 2]));

            gravity = [m1 * obj.g * lg1 * c1 + m2 * obj.g * (l1 * c1 + lg2 * c12)
                m2 * obj.g * lg2 * c12];
            coriolis = [-m2 * l1 * lg2 * s2 .* (2 .* q1d .* q2d + q2d.^2)
                m2 * l1 * lg2 * s2 .* q1d.^2];
            friction = [obj.friction_coeff(1) * q1d
                obj.friction_coeff(2) * q2d];
        end

        %% Kinematics
        function X = getJointsInTaskSpace(obj, state)
        % X = [ x1 y1 x1d y1d x2 y2 x2d y2d ]
            q1 = state(1,:);
            qd1 = state(2,:);
            q2 = state(3,:);
            qd2 = state(4,:);
            xy1 = obj.lengths(1) .* [cos(q1); sin(q1)];
            xy2 = xy1 + obj.lengths(2) .* [cos(q2+q1); sin(q2+q1)];
            
            xy1d = obj.lengths(1) .* [qd1.*cos(q1); -qd1.*sin(q1)];
            xy2d = xy1d + obj.lengths(2) .* [(qd2+q2).*cos(qd1+q1); -(qd2+q2).*sin(qd1+q1)];
            
            X = [xy1; xy1d; xy2; xy2d];
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            line(sum(obj.lengths)*[-1.1 1.1], [0 0], 'LineStyle', '--');
            axis(sum(obj.lengths)*[-1.1 1.1 -1.1 1.1]);

            % Agent handle
            lw = 4.0;
            colors = {[0.1 0.1 0.4], [0.4 0.4 0.8]};
            for i = 1 : 2 : length(obj.lengths)*2
                obj.handleAgent{i} = line([0 0], [0, 0], 'linewidth', lw, 'color', colors{mod((i+1)/2,2)+1});
                obj.handleAgent{i+1} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor',colors{mod((i+1)/2,2)+1});
            end

            % 'Ghost' desired state
            switch obj.mode
                case 'joint'
                    r = 0.1;
                    goalstate = [obj.q_des; obj.qd_des];
                    goalstate = [goalstate(1:2:end); goalstate(2:2:end)];
                    X = obj.getJointsInTaskSpace(goalstate);
                    X = [ zeros(4,1); X ];
                    X(4:4:end,:) = []; % Remove velocities
                    X(3:3:end,:) = [];
                    for i = 1 : 2 : size(X,1) - 2
                        h = line([X(i,:) X(i+2,:)], [X(i+1,:), X(i+3,:)], 'linewidth', lw, 'color', 'r');
                        h.Color(4) = 0.3;
                        h = rectangle('Position',[X(i+2,:)-r,X(i+3,:)-r,2*r,2*r],'Curvature',[1,1],'FaceColor','r');
                        h.EdgeColor(4) = 0.1;
                        h.FaceColor(4) = 0.1;
                    end
                case 'task'
                    r = 0.1;
                    h = rectangle('Position',[obj.endEff_des(1)-r,obj.endEff_des(2)-r,2*r,2*r],'Curvature',[1,1],'FaceColor','r');
                    h.EdgeColor(4) = 0.1;
                    h.FaceColor(4) = 0.1;
                otherwise
                    error('Unknown mode.')
            end
        end
        
        function updateplot(obj, state)
            r = 0.1;
            X = obj.getJointsInTaskSpace(state);
            X = [ zeros(4,1); X ];
            X(4:4:end,:) = []; % Remove velocities
            X(3:3:end,:) = [];
            for i = 1 : 2 : size(X,1) - 2
                obj.handleAgent{i}.XData = [X(i,:) X(i+2,:)];
                obj.handleAgent{i}.YData = [X(i+1,:), X(i+3,:)];
                obj.handleAgent{i+1}.Position = [X(i+2,:)-r,X(i+3,:)-r,2*r,2*r];
            end
            drawnow limitrate
        end
        
        function pixels = render(obj, state)
            n_links = length(obj.lengths);
            meters_to_pixels = 5;
            tot_size = meters_to_pixels*n_links*2+2;
            
            if nargin == 1, pixels = tot_size^2; return, end
            
            n = size(state,2);
            pixels = zeros(tot_size,tot_size,n);
            
            X = obj.getJointsInTaskSpace(state) * meters_to_pixels;
            X = [ zeros(4,n); X ];
            X(4:4:end,:) = []; % Remove velocities
            X(3:3:end,:) = [];
            
            x = X(1:2:end-2,:)';
            y = X(2:2:end-2,:)';
            from = [x(:); y(:)]';
            x = X(3:2:end,:)';
            y = X(4:2:end,:)';
            to = [x(:); y(:)]';
            rows_cols = ceil( linspaceNDim(from(:)',to(:)',meters_to_pixels+1) + tot_size/2 );
            rows = rows_cols(1:end/2,:);
            cols = rows_cols(end/2+1:end,:);
            cols = -cols + tot_size;
            pages = repmat((1:n)',n_links,meters_to_pixels+1);
            pixels(sub2ind(size(pixels), cols, rows, pages)) = 1;
        end
        
    end
    
end