classdef MCarContinuous < MDP
% REFERENCE
% D Ernst, P Geurts, L Wehenkel
% Tree-Based Batch Mode Reinforcement Learning (2005)

    %% Properties
    properties
        % Environment variables
        dt = 0.1;   % Timestep
        mass = 1;   % Mass
        g = 9.81;   % Gravity
        s = 5;      % Slope

        % MDP variables
        dstate = 2;
        daction = 1;
        dreward = 1;
        gamma = 0.95;
        isAveraged = 0;

        % Bounds
        stateLB = [-1 -3]';
        stateUB = [1 3]';
        actionLB = -4;
        actionUB = 4;
        rewardLB = -1;
        rewardUB = 1;
    end
    
    methods

        %% Simulator
        function state = initstate(obj, n)
            state = bsxfun(@plus, bsxfun(@times, ...
                (obj.stateUB - obj.stateLB), rand(obj.dstate,n)), obj.stateLB);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            % Check action
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);

            position = state(1,:);
            velocity = state(2,:);
            throttle = action;

            % Update state
            acceleration = obj.ddp(position, velocity, throttle);
            
            pNext = position + obj.dt * velocity + 0.5 * obj.dt^2 * acceleration;
            vNext = velocity + obj.dt * acceleration;
            nextstate = [pNext; vNext];
            
            % Compute reward
            [reward, absorb] = obj.reward(pNext, vNext);
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
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
        function [reward, absorb] = reward(obj, position, velocity)
            nstates = size(position,2);
            reward = zeros(1,nstates);
            absorb = false(1,nstates);
            fail = (position < obj.stateLB(1)) | ...
                    (velocity > obj.stateUB(2)) | ...
                    (velocity < obj.stateLB(2));
            success = (position > obj.stateUB(1)) & ...
                    (velocity <= obj.stateUB(2)) & ...
                    (velocity >= obj.stateLB(2));
            reward(fail) = -1;
            reward(success) = 1;
            absorb(fail | success) = 1;
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
        end
        
    end
    
end