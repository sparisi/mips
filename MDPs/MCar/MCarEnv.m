classdef (Abstract) MCarEnv < MDP
% REFERENCE
% D Ernst, P Geurts, L Wehenkel
% Tree-Based Batch Mode Reinforcement Learning (2005)

    %% Environment variables
    properties(Constant)
        dt = 0.1;   % Timestep
        mass = 1;   % Mass
        g = 9.81;   % Gravity
        s = 5;      % Slope
    end
    
    methods
        
        %% Transition function
        function nextstate = transition(obj, state, action)
            position = state(1,:);
            velocity = state(2,:);
            acceleration = obj.ddp(position, velocity, action);
            pNext = position + obj.dt * velocity + 0.5 * obj.dt^2 * acceleration;
            vNext = velocity + obj.dt * acceleration;
            vNext = bsxfun(@max, bsxfun(@min,vNext,obj.stateUB(2)), obj.stateLB(2)); % Bound velocity
            nextstate = [pNext; vNext];
        end
        
        %% Dynamics
        function y = hill(obj, x)
        % Height of the car (y-position), given the x-position
            idx = x < 0;
            y(idx) = x(idx).^2 + x(idx);
            y(~idx) = x(~idx) ./ sqrt(1 + obj.s*x(~idx).^2);
        end
        
        function dy = dhill(obj, x)
        % Derivative of the y-position
            idx = x < 0;
            dy(idx) = 2 * x(idx) + 1;
            dy(~idx) = 1 ./ sqrt(1 + obj.s.*x(~idx).^2) ...
                - obj.s.*x(~idx).^2 ./ (1 + obj.s.*x(~idx).^2).^1.5;
        end
        
        function acceleration = ddp(obj, position, velocity, throttle)
            A = throttle ./ ( obj.mass * (1 + obj.dhill(position).^2) );
            B = obj.g * obj.dhill(position) ./ ( 1 + obj.dhill(position).^2 );
            C = velocity.^2 .* obj.dhill(position) .* obj.dhill(obj.dhill(position)) ./ (1 + obj.dhill(position).^2);
            acceleration = A - B - C;
        end
        
        %% Reward function
        function [success, fail] = status(obj, state)
            position = state(1,:);
            fail = position < obj.stateLB(1);
            success = position > obj.stateUB(1);
        end
        
        function reward = reward(obj, state, action, nextstate)
            [success, fail] = obj.status(nextstate);
            reward = zeros(size(success));
            reward(fail) = -1;
            reward(success) = 1;
        end
   
        function absorb = isterminal(obj, state)
            [success, fail] = obj.status(state);
            absorb = fail | success;
        end
        
    end
    
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            xEnv = linspace(obj.stateLB(1),obj.stateUB(1),100)';
            yEnv = obj.hill(xEnv);
            baseline = min(yEnv) - 0.1;
            fill([xEnv(1); xEnv; xEnv(end)],[baseline, yEnv, baseline],'g') % Hill
            plot(xEnv,yEnv,'b','LineWidth',2); % Road
            plot(xEnv(end),yEnv(end),'diamond','MarkerSize',12,'MarkerFaceColor','m') % Goal
            axis([obj.stateLB(1),obj.stateUB(1),baseline,max(yEnv)])
            
            obj.handleAgent = plot(0,0,'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state(1);
            obj.handleAgent.YData = obj.hill(state(1));
            drawnow limitrate
        end
        
    end
    
end