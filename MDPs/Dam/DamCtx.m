classdef DamCtx < CMDP
    
    %% Properties
    properties
        % Environment variables
        DAM_INFLOW_STD = 1e-8;
        Q_MEF = 0.0;
        GAMMA_H2O = 1000.0;
        G = 9.81;
        
        % Contextual variables and bounds
        dctx = 8;
        ctxLB = [80 % S_MIN_REL
            20      % DAM_INFLOW_MEAN
            1       % S
            40      % H_FLO_U
            0.1     % ETA
            2       % W_HYD
            30      % W_IRR
            10];    % Q_FLO_D
        ctxUB = [120 80 2 60 1 6 80 50]';

        % MDP variables
        dstate = 1;
        daction = 1;
        dreward = 1;
        isAveraged = 1;
        gamma = 1;
        
        % Bounds
        stateLB = 0;
        stateUB = inf;
        actionLB = -inf;
        actionUB = inf;
        rewardLB = -inf;
        rewardUB = 0;
    end
    
    methods
        
        %% Simulator
        function ctx = getcontext(obj, n)
            ctx = myunifrnd(obj.ctxLB, obj.ctxUB, n);
        end
        
        function state = initstate(obj, n)
            state = myunifrnd(0,300,n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action, context)
            nstates = size(state,2);
            reward = zeros(obj.dreward,nstates);
            
            % Bound the action
            actionLB = max(bsxfun(@minus,state,context(1,:)), 0);
            actionUB = state;
            
            % Penalty proportional to the violation
            bounded_action = min(max(action,actionLB),actionUB);
            penalty = -abs(bounded_action - action);
            
            % Transition dynamic
            action = bounded_action;
            dam_inflow = mymvnrnd(context(2,:), obj.DAM_INFLOW_STD^2, nstates);
            nextstate = state + dam_inflow - action;
            
            % Cost due to the excess level w.r.t. a flooding threshold (upstream)
            reward(1,:) = -max(nextstate./context(3,:) - context(4,:), 0);
            
            % Deficit in the water supply w.r.t. the water demand
            reward(2,:) = -max(context(7,:) - action, 0);
            
            q = max(action - obj.Q_MEF, 0);
            p_hyd = context(5,:) .* obj.G .* obj.GAMMA_H2O .* nextstate ./ context(3,:) .* q ./ (3.6e6);
            
            % Deficit in the hydroelectric supply w.r.t. the hydroelectric demand
            reward(3,:) = -max(context(6,:) - p_hyd, 0);
            
%             reward = sum(reward,1) + penalty;
            reward = reward(1,:) + reward(3,:) + penalty;
            absorb = false(1,nstates);

            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
    end
        
    %% Plotting
    methods(Hidden = true)

        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            x = sqrt(obj.S); y = x; % Assume square surface
            vertices = @(z1,z2)[x y z1; % Cube vertices
                0 y z1;
                0 y z2;
                x y z2;
                0 0 z2;
                x 0 z2;
                x 0 z1;
                0 0 z1];
            faces = [1 2 3 4; % Cube faces
                4 3 5 6;
                6 7 8 5;
                1 2 8 7;
                6 7 1 4;
                2 3 5 8];
            
            patch('Faces', faces, 'Vertices', vertices(obj.H_FLO_U-1,obj.H_FLO_U+1), 'FaceColor', 'r'); % Flooding threshold
            obj.handleAgent = patch('Faces', faces, 'Vertices', vertices(obj.H_FLO_U-1,obj.H_FLO_U+1), 'FaceColor', 'b'); % Current mass

            axis([0, 1, 0, 1, 0, 200])
            view(3)
        end
        
        function updateplot(obj, state)
            x = sqrt(obj.S); y = x; % Assume square surface
            vertices = @(z1,z2)[x y z1; % Cube vertices
                0 y z1;
                0 y z2;
                x y z2;
                0 0 z2;
                x 0 z2;
                x 0 z1;
                0 0 z1];
            z = state / obj.S;
            obj.handleAgent.Vertices = vertices(0,z);
            drawnow limitrate
        end
        
    end
    
end