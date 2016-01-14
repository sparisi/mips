classdef BicycleBalance < MDP
% REFERENCE
% J Randlov, P Alstrom
% Learning to Drive a Bicycle using Reinforcement Learning and Shaping
% (1998)
%
% Original C code: http://www.ailab.si/dorian/MLControl/BikeCCode.htm
    
    %% Properties
    properties
        % MDP variables
        dstate = 4;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.99;

        % Bounds : state = (theta, theta_dot, omega, omega_dot)
        stateLB = [-1.3963 -inf -pi/15 -inf]';
        stateUB = [1.3963 inf pi/15 inf]';
        actionLB = 1;
        actionUB = 9;
        rewardLB = -1;
        rewardUB = 1;
    end
    
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
        M = (BicycleBalance.Mc + BicycleBalance.Mp);
        sigma_dot = BicycleBalance.v / BicycleBalance.R;
        I_bike = ((13.0/3)*BicycleBalance.Mc*BicycleBalance.h^2 + BicycleBalance.Mp*(BicycleBalance.h+BicycleBalance.dCM)^2);
        I_dc = (BicycleBalance.Md*BicycleBalance.R^2);
        I_dv = ((3.0/2)*BicycleBalance.Md*BicycleBalance.R^2);
        I_dl = ((1.0/2)*BicycleBalance.Md*BicycleBalance.R^2);
        l = 1.11; % distance between the point where the front and back tyre touch the ground
        maxNoise = 0.02; % max noise in the displacement
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj,n)
            state = repmat([0 0 0 0]',1,n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            % Parse input
            allActions = [-2 -2 -2 0 0 0 2 2 2
                -0.02 0 0.02 -0.02 0 0.02 -0.02 0 0.02];
            T = allActions(1,action); % torque
            d = allActions(2,action); % displacement
            
            theta     = state(1,:);
            theta_dot = state(1,:);
            omega     = state(3,:);
            omega_dot = state(4,:);
            
            nstates = size(state,2);
            
            % Noise in displacement
            noise = (rand(1,nstates) * 2 - 1) * obj.maxNoise;
            d = d + noise;
            
            old_omega = omega;
            
            rCM = 9999999*ones(1,nstates); % just a large number (to avoid dividing by zero)
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
            
            idx = abs(theta) > 1.3963; % handlebars cannot turn more than 80 degrees
            theta(idx) = sign(theta(idx)) * 1.3963;
            
            nextstate(1,:) = theta;
            nextstate(2,:) = theta_dot;
            nextstate(3,:) = omega;
            nextstate(4,:) = omega_dot;
            
            % Compute reward
            reward = (old_omega / obj.stateLB(3)).^2 - (omega / obj.stateLB(3)).^2;
            
            % Check for terminal condition
            idx = ( omega < obj.stateLB(3) ) | ( omega > obj.stateUB(3) ); % the bike has fallen over
            absorb = false(1,nstates);
            absorb(idx) = true;

            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            warning('This MDP does not support plotting!')
        end
        
        function updateplot(obj, state)
            warning('This MDP does not support plotting!')
        end
        
    end
    
end