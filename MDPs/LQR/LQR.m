classdef LQR < MOMDP
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)
    
    %% Properties
    properties
        % Environment variables
        e = 0.1;
        A
        B
        x0
        Q
        R
        
        % MDP variables
        dstate
        daction
        dreward
        isAveraged = 0;
        gamma = 0.9;
        
        % Upper/Lower Bounds
        stateLB
        stateUB
        actionLB
        actionUB
        rewardLB
        rewardUB
        
        % Multiobjective
        utopia
        antiutopia
    end
    
    methods

        %% Constructor
        function obj = LQR(dim)
            obj.A = eye(dim);
            obj.B = eye(dim);
            obj.x0 = 10*ones(dim,1);
            obj.Q = repmat({obj.e*eye(dim)},dim,1);
            obj.R = repmat({(1-obj.e)*eye(dim)},dim,1);
            for i = 1 : dim
                obj.Q{i}(i,i) = 1-obj.e;
                obj.R{i}(i,i) = obj.e;
            end
            
            obj.dstate = dim;
            obj.daction = dim;
            obj.dreward = dim;
            
            % Bounds
            obj.stateLB = -inf(dim,1);
            obj.stateUB = inf(dim,1);
            obj.actionLB = -inf(dim,1);
            obj.actionUB = inf(dim,1);
            obj.rewardLB = -inf(dim,1);
            obj.rewardUB = zeros(dim,1);
            
            % Multiobjective
            switch dim
                case 2
                    obj.utopia = -150*ones(1,dim);
                    obj.antiutopia = -310*ones(1,dim);
                case 3
                    obj.utopia = -195*ones(1,dim);
                    obj.antiutopia = -360*ones(1,dim);
                case 5
                    obj.utopia = -283*ones(1,dim);
                    obj.antiutopia = -436*ones(1,dim);
                otherwise
                    warning('Multiobjective framework not available for this number of objective.')
            end
        end
        
        %% Simulator
        function state = initstate(obj, n)
            state = repmat(obj.x0,1,n);
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstate = size(state,2);
            absorb = false(1,nstate);
            nextstate = obj.A*state + obj.B*action;
            reward = zeros(obj.dreward,nstate);
            for i = 1 : obj.dreward
                reward(i,:) = -diag((state'*obj.Q{i}*state + action'*obj.R{i}*action));
            end
        end
        
        %% Multiobjective
        function [front, weights] = truefront(obj)
            try
                front = dlmread(['lqr_front' num2str(obj.dreward) 'd.dat']);
                weights = dlmread(['lqr_w' num2str(obj.dreward) 'd.dat']);
                warning('This frontier was obtained with a Gaussian policy with fixed identity covariance!')
            catch
                warning('Multiobjective framework not available for this number of objective.')
            end
        end

        function fig = plotfront(obj, front, fig)
            front = sortrows(front);
            if nargin == 2, fig = figure(); end
            hold all
            
            if obj.dreward == 2
                plot(front(:,1),front(:,2),'+')
            elseif obj.dreward == 3
                plot3(front(:,1),front(:,2),front(:,3),'o')
                box on
            else
                warning('Can plot only 2 and 3 dimensions.')
                return
            end
            
            xlabel 'Obj 1'
            ylabel 'Obj 2'
            zlabel 'Obj 3'
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