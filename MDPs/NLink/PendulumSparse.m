classdef PendulumSparse < NLinkEnv
% REFERENCE
% https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
% 
% Difference from the classical pendulum. The penalty on the action and
% velocity is kept, but the reward depending on the angle (distance from
% the upright position) is given only when the pendulum is within 25
% degrees from the top. Everywhere else, the agent receives always -10.

    %% Properties
    properties
        % Environment variables
        lengths = 1;
        masses = 1;
        g = 10;
        dt = 0.05;
        
        q_des = 0; % Upright position
        
        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds : state = [q qd]
        stateLB = [-pi, -8]';
        stateUB = [pi, 8]';
        actionLB = -2;
        actionUB = 2;
        rewardLB = - pi.^2 - 0.1*8.^2 - 0.001*2.^2;
        rewardUB = 0;
    end
    
    methods

        %% Simulation
        function state = init(obj, n)
            if nargin == 1, n = 1; end
            state = repmat([-pi 0]',1,n);
%             state = myunifrnd([-pi, -1], [pi, 1], n);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            q = state(1,:);
            qd = state(2,:);
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
            close_to_top = abs(angdiff(q,obj.q_des,'rad')) < 0.5;
            reward = zeros(1, size(state,2));
            reward_angle = - angdiff(q,obj.q_des,'rad').^2;
            reward_dense = - 0.1*qd.^2 - 0.001*action.^2;
            reward(~close_to_top) = reward_dense(~close_to_top) - 10;
            reward(close_to_top) = reward_dense(close_to_top) + reward_angle(close_to_top);
            qd = qd + ...
                (-3*obj.g/(2*obj.lengths).*sin(q+pi) + ...
                3./(obj.masses*obj.lengths.^2).*action) * obj.dt;
            q = q + qd*obj.dt;
            q = wrapinpi(q);
            qd = bsxfun(@max, bsxfun(@min,qd,obj.stateUB(2)), obj.stateLB(2));
            nextstate = [q; qd];
            absorb = false(1,size(state,2));
            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Kinematics
        function X = getJointsInTaskSpace(obj, state)
%             X = [ x y xd yd ]
            q = state(1,:);
            qd = state(2,:);
            xy = obj.lengths(1) .* [cos(q); sin(q)];
            xy1 = obj.lengths(1) .* [qd.*cos(q); -qd.*sin(q)];
            X = [xy; xy1];
        end

        %% Plotting
        function initplot(obj)
            initplot@NLinkEnv(obj)
%             text(1,0.15,'0 (2\pi)','HorizontalAlignment','center')
%             text(-1,0.15,'\pi (-\pi)','HorizontalAlignment','center')
%             text(0,1,'\pi/2','HorizontalAlignment','center')
%             text(0,-1,'-\pi/2','HorizontalAlignment','center')
            pbaspect([1 1 1])
%             rectangle('Position',[-1,-1,2,2],'Curvature',[1,1]);
        end
        
        function updateplot(obj, state)
            % Upright position is at pi/2. In OpenAI gym, it is at 0.
            % Since this code follows gym equations, for plotting we need 
            % to shift everything by pi/2.
            state(1,:) = state(1,:) + pi/2;
            updateplot@NLinkEnv(obj,state);
        end
        
        function [pixels, clims, cmap] = render(obj, state)
            % See above.
            state(1,:) = state(1,:) + pi/2;
            [pixels, clims, cmap] = render@NLinkEnv(obj,state);
        end
    end
     
end
