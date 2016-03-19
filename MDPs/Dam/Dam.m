classdef Dam < MOMDP
% REFERENCE
% A Castelletti, F Pianosi, M Restelli
% Tree-based Fitted Q-Iterationfor Multi-Objective Markov Decision Problems
% processes in water resource management (2012)
    
    %% Properties
    properties
        % Environment variables
        S = 1.0; % Reservoir surface
        W_IRR = 50.0; % Water demand
        H_FLO_U = 50.0; % Flooding threshold (upstream, i.e., height of the dam)
        S_MIN_REL = 100.0; % Release threshold (i.e., max capacity)
        DAM_INFLOW_MEAN = 40; % Random inflow (e.g. rain)
        DAM_INFLOW_STD = 10;
        Q_MEF = 0.0;
        GAMMA_H2O = 1000.0; % Water density
        W_HYD = 4.36; % Hydroelectric demand
        Q_FLO_D = 30.0; % Flooding threshold (downstream, i.e., releasing too much water)
        ETA = 1.0; % Turbine efficiency
        G = 9.81; % Gravity
        
        % Initial states
        s_init = [9.6855361e+01, 5.8046026e+01, ...
            1.1615767e+02, 2.0164311e+01, ...
            7.9191000e+01, 1.4013098e+02, ...
            1.3101816e+02, 4.4351321e+01, ...
            1.3185943e+01, 7.3508622e+01, ...
            ];
        
        % MDP variables
        dstate = 1;
        daction = 1;
        dreward           % Change according to the number of objectives
        isAveraged = 1;
        gamma = 1;
        
        % Bounds
        stateLB = 0;
        stateUB = inf;
        actionLB = -inf;
        actionUB = inf;
        rewardLB = -[inf inf inf inf]';
        rewardUB = [0 0 0 0]';

        % Multiobjective
        utopia
        antiutopia
    end
    
    properties
        %%
        penalize = 0; % 1 to penalize the policy when it violates the problem's constraints
    end
    
    methods
        
        %% Constructor
        function obj = Dam(dim)
            obj.dreward = dim;
            switch dim
                case 2
                    obj.utopia = [-0.5, -9];
                    obj.antiutopia = [-2.5, -11];
                case 3
                    obj.utopia = [-0.5, -9, -0.001];
                    obj.antiutopia = [-65, -12, -0.7];
                case 4
                    obj.utopia = [-0.5, -9, -0.001, -9];
                    obj.antiutopia = [-65, -12, -0.7, -12];
                otherwise
                    error('Wrong number of rewards.')
            end
        end
        
        %% Simulator
        function state = initstate(obj, n)
            if ~obj.penalize
                idx = randi(length(obj.s_init), [1,n]);
                state = obj.s_init(idx);
            else
                state = myunifrnd(0,160,n);
            end
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstates = size(state,2);
            reward = zeros(obj.dreward,nstates);
            
            % Bound the action
            actionLB = max(bsxfun(@minus,state,obj.S_MIN_REL), 0);
            actionUB = state;
            
            % Penalty proportional to the violation
            bounded_action = min(max(action,actionLB),actionUB);
            penalty = - obj.penalize * abs(bounded_action - action);
            
            % Transition dynamic
            action = bounded_action;
            dam_inflow = mymvnrnd(obj.DAM_INFLOW_MEAN, obj.DAM_INFLOW_STD^2, nstates);
            nextstate = max(state + dam_inflow - action, 0); % There is a very small chance that dam_inflow < 0
            
            % Cost due to the excess level w.r.t. a flooding threshold (upstream)
            reward(1,:) = -max(nextstate/obj.S - obj.H_FLO_U, 0) + penalty;
            
            % Deficit in the water supply w.r.t. the water demand
            reward(2,:) = -max(obj.W_IRR - action, 0) + penalty;
            
            q = max(action - obj.Q_MEF, 0);
            p_hyd = obj.ETA .* obj.G .* obj.GAMMA_H2O .* nextstate ./ obj.S .* q ./ (3.6e6);
            
            % Deficit in the hydroelectric supply w.r.t. the hydroelectric demand
            reward(3,:) = -max(obj.W_HYD - p_hyd, 0) + penalty;
            
            % Cost due to the excess level w.r.t. a flooding threshold (downstream)
            reward(4,:) = -max(action - obj.Q_FLO_D, 0) + penalty;
            
            reward = reward(1:obj.dreward,:);
            absorb = false(1,nstates);

            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Multiobjective
        function [front, weights] = truefront(obj)
            front = dlmread(['dam_front' num2str(obj.dreward) 'd.dat']);
            weights = dlmread(['dam_w' num2str(obj.dreward) 'd.dat']);
        end

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(obj, front, varargin{:});
            xlabel 'Flooding'
            ylabel 'Water Demand'
            zlabel 'Hydroelectric Demand'
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