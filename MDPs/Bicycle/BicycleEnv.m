classdef (Abstract) BicycleEnv < MDP
% REFERENCE
% J Randlov, P Alstrom
% Learning to Drive a Bicycle using Reinforcement Learning and Shaping
% (1998)
%
% Original C code: http://www.ailab.si/dorian/MLControl/BikeCCode.htm
    
    %% Environment variables
    properties(Constant)
        dt = 0.01;
        v = (10.0/3.6); % 10 km/h in m/s
        g = 9.82;
        dCM = 0.3;
        c = 0.66;
        h = 0.94;
        Mc = 15.0;
        Md = 1.7;
        Mp = 60.0;
        R = 0.34; % tyre radius
        M = (BicycleEnv.Mc + BicycleEnv.Mp);
        sigma_dot = BicycleEnv.v / BicycleEnv.R;
        I_bike = ((13.0/3)*BicycleEnv.Mc*BicycleEnv.h^2 + BicycleEnv.Mp*(BicycleEnv.h+BicycleEnv.dCM)^2);
        I_dc = (BicycleEnv.Md*BicycleEnv.R^2);
        I_dv = ((3.0/2)*BicycleEnv.Md*BicycleEnv.R^2);
        I_dl = ((1.0/2)*BicycleEnv.Md*BicycleEnv.R^2);
        l = 1.11; % distance between the point where the front and back tyre touch the ground
        maxNoise = 0.02; % max noise in the displacement
    end
    
    methods
        
        function nextstate = transition(obj, state, action)
            [dstate, nstates] = size(state);
 
            T = action(1,:); % Torque
            d = action(2,:); % Displacement
            
            theta     = state(1,:);
            theta_dot = state(1,:);
            omega     = state(3,:);
            omega_dot = state(4,:);
            
            rCM = 9999999*ones(1,nstates); % Just a large number (to avoid dividing by zero)
            rf = 9999999*ones(1,nstates);
            rb = 9999999*ones(1,nstates);
            
            idx = theta ~= 0;
            rCM(idx) = sqrt( (obj.l-obj.c).^2 + obj.l.^2 ./ tan(theta(idx)).^2 );
            rf(idx) = obj.l ./ abs(sin(theta(idx)));
            rb(idx) = obj.l ./ abs(tan(theta(idx))); % rCM, rf and rb are always positive
            
            % Main physics equations
            phi = omega + atan(d / obj.h);
            omega_d_dot = ( obj.h .* obj.M .* obj.g .* sin(phi) ...
                - cos(phi) .* (obj.I_dc .* obj.sigma_dot .* theta_dot ...
                + sign(theta) .* obj.v^2 .* (obj.Md * obj.R .* (1.0 ./ rf + 1.0 ./ rb) ...
                + obj.M .* obj.h ./ rCM) ) ...
                ) ./ obj.I_bike;
            theta_d_dot = (T - obj.I_dv .* omega_dot .* obj.sigma_dot) ./ obj.I_dl;
            
            % State transition
            omega_dot = omega_dot + omega_d_dot * obj.dt;
            omega     = omega     + omega_dot   * obj.dt;
            theta_dot = theta_dot + theta_d_dot * obj.dt;
            theta     = theta     + theta_dot   * obj.dt;
            theta     = wrapinpi(theta);
            
            idx = abs(theta) > 1.3963; % handlebars cannot turn more than 80 degrees
            theta(idx) = sign(theta(idx)) * 1.3963;
            
            nextstate = [theta;
                theta_dot;
                omega;
                omega_dot];
            
            % Next part is for the balancing problem
            if dstate == 4, return, end

            psi       = state(5,:);
            xf        = state(6,:);
            yf        = state(7,:);
            xb        = state(8,:);
            yb        = state(9,:);            
            
            % Update front tyre position
            temp = obj.v * obj.dt * 0.5 ./ rf;
            idx = temp > 1;
            temp(idx) = sign(psi(idx) + theta(idx)) * 0.5 * pi;
            temp(~idx) = sign(psi(~idx) + theta(~idx)) .* asin(temp(~idx));
            xf = xf + obj.v * obj.dt .* ( -sin(psi + theta + temp) );
            yf = yf + obj.v * obj.dt .* ( cos(psi + theta + temp) );

            % Update back tyre position
            temp = obj.v * obj.dt * 0.5 ./ rb;
            idx = temp > 1;
            temp(idx) = sign(psi(idx)) * 0.5 * pi;
            temp(~idx) = sign(psi(~idx)) .* asin(temp(~idx));
            xb = xb + obj.v * obj.dt .* ( -sin(psi + temp) );
            yb = yb + obj.v * obj.dt .* ( cos(psi + temp) );

            % Preventing numerical drift
            wheelbase = sqrt((xf - xb).^2 + (yf - yb).^2);
            idx = abs(wheelbase - obj.l) > 0.01;
            relative_error = obj.l ./ wheelbase(idx) - 1.0;
            xb(idx) = xb(idx) + (xb(idx) - xf(idx)) .* relative_error;
            yb(idx) = yb(idx) + (yb(idx) - yf(idx)) .* relative_error;

            % Update heading
            psi = zeros(1,nstates);
            delta_y = yf - yb;
            idx = (xf == xb) & (delta_y < 0.0);
            psi(idx) = pi;
            idx = ~idx & delta_y > 0.0;
            psi(idx) = atan((xb(idx) - xf(idx)) ./ delta_y(idx));
            psi(~idx) = sign(xb(~idx) - xf(~idx)) * 0.5 * pi - atan(delta_y(~idx) ./ (xb(~idx) - xf(~idx)));
            
            nextstate = [nextstate;
                psi;
                xf;
                yf;
                xb;
                yb];
        end            
        
        %% Reward functions for the balancing problem
        function reward = rewardRandlovCustom(obj, nextstate)
        % Reward function used in the original implementation. See the code 
        % linked at the beginning of the class for more info.
            xf        = nextstate(6,:);
            yf        = nextstate(7,:);
            xb        = nextstate(8,:);
            yb        = nextstate(9,:);

            psi_goal = (xf-xb) .* (obj.goal(1)-xf) + (yf-yb) .* (obj.goal(2)-yf);
            scalar = psi_goal ./ ( obj.l .* sqrt( (obj.goal(1)-xf).^2 + (obj.goal(2)-yf).^2 ) );
            tvaer = (-yf+yb) .* (obj.goal(1)-xf) + (xf-xb) .* (obj.goal(2)-yf);

            idx = tvaer <= 0;
            psi_goal = zeros(1,length(xf));
            psi_goal(idx) = scalar(idx) - 1;
            psi_goal(~idx) = abs(scalar(~idx) - 1);

            reward = (0.95 - psi_goal.^2) * 0.0001;
        end         
        
        function reward = rewardRandlov(obj, nextstate)
        % Reward function defined in the original paper.
            xf = nextstate(6,:);
            yf = nextstate(7,:);
            xb = nextstate(8,:);
            yb = nextstate(9,:);

            v1 = [xf-xb; yf-yb];
            v2 = [obj.goal(1)-xb; obj.goal(2)-yb];
            psi_goal = acos(dot(v1,v2)./(matrixnorms(v1,2).*matrixnorms(v2,2)));

            reward = (4 - psi_goal.^2) * 0.00004;
        end
        
        function reward = rewardErnst(obj, state, nextstate)
        % Reward function defined in "Tree-Based Batch Mode Reinforcement 
        % Learning (2005)" by Ernst et al.
            xf1 = state(6,:);
            yf1 = state(7,:);
            xb1 = state(8,:);
            yb1 = state(9,:);
            xf2 = nextstate(6,:);
            yf2 = nextstate(7,:);
            xb2 = nextstate(8,:);
            yb2 = nextstate(9,:);

            v1 = [xf1-xb1; yf1-yb1];
            v2 = [obj.goal(1)-xb1; obj.goal(2)-yb1];
            psi_goal1 = acos(dot(v1,v2)./(matrixnorms(v1,2).*matrixnorms(v2,2)));
            v1 = [xf2-xb2; yf2-yb2];
            v2 = [obj.goal(1)-xb2; obj.goal(2)-yb2];
            psi_goal2 = acos(dot(v1,v2)./(matrixnorms(v1,2).*matrixnorms(v2,2)));

            reward = 0.1 * (abs(psi_goal1) - abs(psi_goal2));
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            obj.handleAgent{1} = plotCircle3D([0 obj.l obj.R], [1 0 0], obj.R, 'r', 1); % Front tyre
            obj.handleAgent{1}.EdgeColor = 'r';
            obj.handleAgent{2} = plotCircle3D([0 0 obj.R], [1 0 0], obj.R, 'b', 1); % Back tyre
            obj.handleAgent{2}.EdgeColor = 'b';
            obj.handleAgent{3} = plot3([0 0], [0 obj.l], [obj.R obj.R], 'k', 'LineWidth', 0.8); % Bicycle frame
            obj.handleAgent{4} = plot(0,obj.l,'k'); % Trace of the front tyre
            box on
            xlabel x, ylabel y, zlabel z
            view(35,75)
%             rotate3d on
        end
        
        function updateplot(obj, state)
            theta = state(1,:);
            omega = state(3,:);
            
            if size(state,1) > 4
                psi   = state(5,:);
                xf    = state(6,:);
                yf    = state(7,:);
                xb    = state(8,:);
                yb    = state(9,:);
            else
                psi = 0;
                xf = 0;
                yf = obj.l;
                xb = 0;
                yb = 0;
                view(0,90)
            end

            x = 0 : 0.1 : 2*pi;

            % Plot back tyre
            normal_back = [cos(psi) sin(psi) -sin(omega)];
            center_back = [xb yb obj.R*cos(omega)];
            vec = null(normal_back);
            points = repmat(center_back',1,size(x,2)) ...
                + obj.R * (vec(:,1) * cos(x) ...
                + vec(:,2) * sin(x));
            obj.handleAgent{2}.XData = points(1,:);
            obj.handleAgent{2}.YData = points(2,:);
            obj.handleAgent{2}.ZData = points(3,:);
            
            % Plot front tyre
            normal_front = [cos(theta + psi) sin(theta + psi) -sin(omega)];
            center_front = [xf yf obj.R*cos(omega)];
            vec = null(normal_front);
            points = repmat(center_front',1,size(x,2)) ...
                + obj.R * (vec(:,1) * cos(x) ...
                + vec(:,2) * sin(x));
            obj.handleAgent{1}.XData = points(1,:);
            obj.handleAgent{1}.YData = points(2,:);
            obj.handleAgent{1}.ZData = points(3,:);
            
            % Plot bicycle frame
            obj.handleAgent{3}.XData = [center_front(1) center_back(1)];
            obj.handleAgent{3}.YData = [center_front(2) center_back(2)];
            obj.handleAgent{3}.ZData = [center_front(3) center_back(3)];

            % Update trace
            obj.handleAgent{4}.XData(end+1) = xf;
            obj.handleAgent{4}.YData(end+1) = yf;
            
            axis([xf-2 xf+2 yf-2 yf+2])

            drawnow limitrate
        end
        
    end
    
end