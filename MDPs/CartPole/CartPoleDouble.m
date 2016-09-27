classdef CartPoleDouble < MDP
% REFERENCE
% A P Wieland
% Evolving Controls for Unstable Systems (1991)
    
    %% Properties
    properties
        % Environment variables
        g = 9.8;
        masscart = 1.0;
        masspole = [0.1 0.01]';
        length = [0.5 0.05]'; % Actually distance from the pivot to the poles centre of mass (so full length is the double)
        force = 10.0;
        dt = 0.02;
        mu_c = 0.0005; % Coefficient of friction of cart on track
        mu_p = 0.000002; % Coefficient of friction of the pole's hinge
        
        % Finite actions
        allactions = [1 2];

        % MDP variables
        dstate = 6;
        daction = 1;
        dreward = 1;
        isAveraged = 0;
        gamma = 0.9;

        % Bounds : state = [x xd theta thetad]
        stateLB = [-2.4, -inf, -deg2rad(15), -deg2rad(15) -inf, -inf]';
        stateUB = [2.4, inf, deg2rad(15), deg2rad(15), inf, inf]';
        actionLB = 1;
        actionUB = 2;
        rewardLB = -1;
        rewardUB = 0;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj,n)
            initLB = [-2.4, -1, -deg2rad(15), -deg2rad(15) -1, -1]';
            initUB = [2.4, 1, deg2rad(15), deg2rad(15), 1, 1]';
            state = bsxfun(@plus, ...
                bsxfun(@times, (initUB - initLB), rand(obj.dstate,n)), initLB);
            state(1:2,:) = 0; % the cart is always in the middle, with 0 vel
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            forces = [-obj.force obj.force];
            F = forces(action);
            
            x = state(1,:);
            xd = state(2,:);
            theta = state(3:4,:);
            thetad = state(5:6,:);
            
            costheta = cos(theta);
            sintheta = sin(theta);
            
            polemass_length = obj.length .* obj.masspole;
            temp = obj.mu_p .* bsxfun(@times, thetad, 1./ polemass_length);
            F_tilde = - bsxfun(@times, obj.masspole .* obj.length, thetad.^2) .* sintheta ...
                + 0.75 .* bsxfun(@times, obj.masspole, costheta) .* (obj.g .* sintheta - temp);
            mass_tilde = bsxfun(@times, obj.masspole, (1 - 0.75 .* costheta.^2));
            
            xdd = ( F - obj.mu_c .* sign(xd) + sum(F_tilde,1) ) ./ (obj.masscart + sum(mass_tilde,1));
            thetadd = 0.75 .* bsxfun( @times, ( bsxfun(@times, xdd, costheta) + obj.g .* sintheta + temp ), 1 ./ obj.length );
            
            x = x + obj.dt .* xd;
            xd = xd + obj.dt .* xdd;
            theta = theta + obj.dt .* thetad;
            theta = wrapinpi(theta); % theta in [-pi, pi]
            thetad = thetad + obj.dt .* thetadd;

            nextstate = [x; xd; theta; thetad];
            
            nstates = size(state,2);
            absorb = false(1,nstates);
            reward = zeros(1,nstates);
            fallen = max(bsxfun(@lt, nextstate, obj.stateLB),[],1) | ...
                max(bsxfun(@gt, nextstate, obj.stateUB),[],1);
            reward(fallen) = -1;
            absorb(fallen) = true;
            
            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            obj.handleAgent{1} = plot(0,0,'k','LineWidth',6); % Cart
            obj.handleAgent{2} = plot(0,0,'ko','MarkerSize',12,'MarkerEdgeColor','k','MarkerFaceColor','k'); % Cart-Pole Link / Wheel
            obj.handleAgent{3} = plot(0,0,'k','LineWidth',4); % Pole1
            obj.handleAgent{4} = plot(0,0,'g','LineWidth',4); % Pole2
            
            plot([obj.stateLB(1),obj.stateLB(1)],[0,10],'r','LineWidth',2)
            plot([obj.stateUB(1),obj.stateUB(1)],[0,10],'r','LineWidth',2)

            axis([-3 3 0 1.5]);
        end
        
        function updateplot(obj, state)
            x1 = state(1);
            y1 = 0.1;
            theta = state(3:4);
            x2 = -sin(theta) .* 2 .* obj.length + x1;
            y2 = cos(theta) .* 2 .* obj.length + y1;
            
            obj.handleAgent{1}.XData = [x1-0.2 x1+0.2];
            obj.handleAgent{1}.YData = [y1 y1];

            obj.handleAgent{2}.XData = x1;
            obj.handleAgent{2}.YData = y1;

            obj.handleAgent{3}.XData = [x1 x2(1)];
            obj.handleAgent{3}.YData = [y1 y2(1)];
            obj.handleAgent{4}.XData = [x1 x2(2)];
            obj.handleAgent{4}.YData = [y1 y2(2)];
            drawnow limitrate
        end
        
    end
    
end