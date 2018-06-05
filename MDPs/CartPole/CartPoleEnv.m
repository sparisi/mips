classdef (Abstract) CartPoleEnv < MDP
% REFERENCE
% A P Wieland
% Evolving Controls for Unstable Systems (1991)
    
    %% Environment variables
    properties(Constant)
        g = 9.81;
        masscart = 1.0;
        masspole = [0.1 0.05]';
        length = [0.5 0.25]'; % Distance from the pivot to the poles centre of mass (so full length is the double)
        force = 10.0;
        dt = 0.05;
        mu_c = 0.0005; % Coefficient of friction of cart on track
        mu_p = 0.000002; % Coefficient of friction of the pole's hinge
    end
    
    %% Simulator
    methods
        function nextstate = transition(obj, state, action)
            nlink = (size(state,1)-2)/2;
            masspoles = obj.masspole(1:nlink);
            lengths = obj.length(1:nlink);
            
            x = state(1,:);
            xd = state(2,:);
            theta = state(3:3+nlink-1,:);
            thetad = state(3+nlink:end,:);
            
            costheta = cos(theta);
            sintheta = sin(theta);

            polemass_length = lengths .* masspoles;
            temp = obj.mu_p .* bsxfun(@times, thetad, 1./ polemass_length);
            F_tilde = - bsxfun(@times, masspoles .* lengths, thetad.^2) .* sintheta ...
                + 0.75 .* bsxfun(@times, masspoles, costheta) .* (obj.g .* sintheta - temp);
            mass_tilde = bsxfun(@times, masspoles, (1 - 0.75 .* costheta.^2));
            
            xdd = ( action - obj.mu_c .* sign(xd) + sum(F_tilde,1) ) ./ (obj.masscart + sum(mass_tilde,1));
            thetadd = 0.75 .* bsxfun( @times, ( bsxfun(@times, xdd, costheta) + obj.g .* sintheta + temp ), 1 ./ lengths );

            x = x + obj.dt .* xd;
            xd = xd + obj.dt .* xdd;
            theta = theta + obj.dt .* thetad;
            theta = wrapinpi(theta); % theta in [-pi, pi]
            thetad = thetad + obj.dt .* thetadd;

            nextstate = [x; xd; theta; thetad];
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
            
            plot([obj.stateLB(1),obj.stateLB(1)],[-10,10],'r','LineWidth',2)
            plot([obj.stateUB(1),obj.stateUB(1)],[-10,10],'r','LineWidth',2)

            axis([-3 3 -1.5 1.5]);
        end
        
        function updateplot(obj, state)
            nlink = (size(state,1)-2)/2;
            x1 = state(1);
            y1 = 0.1;
            theta = state(3:3+nlink-1);
            x2 = -sin(theta) .* 2 .* obj.length(1:nlink) + x1;
            y2 = cos(theta) .* 2 .* obj.length(1:nlink) + y1;
            
            obj.handleAgent{1}.XData = [x1-0.2 x1+0.2];
            obj.handleAgent{1}.YData = [y1 y1];
            obj.handleAgent{2}.XData = x1;
            obj.handleAgent{2}.YData = y1;

            for i = 3 : 3 + nlink - 1
                obj.handleAgent{i}.XData = [x1 x2(i-2)];
                obj.handleAgent{i}.YData = [y1 y2(i-2)];
            end
            drawnow limitrate
        end
    end
    
end