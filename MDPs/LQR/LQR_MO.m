classdef LQR_MO < MOMDP & LQREnv
% Multi-objective version of LQR.
%
% REFERENCE
% S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli 
% Policy gradient approaches for multi-objective sequential decision making
% (2014)
    
    %% Properties
    properties
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
        function obj = LQR_MO(dim)
            e = 0.1;
            obj.A = eye(dim);
            obj.B = eye(dim);
            obj.x0 = 10*ones(dim,1);
            obj.Q = repmat(e*eye(dim),1,1,dim);
            obj.R = repmat((1-e)*eye(dim),1,1,dim);
            for i = 1 : dim
                obj.Q(i,i,i) = 1-e;
                obj.R(i,i,i) = e;
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
                    warning('Multiobjective framework not available for the desired number of objective.')
            end
        end
        
        %% Simulator
        function state = init(obj, n)
            state = repmat(obj.x0,1,n); % Fixed
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstate = size(state,2);
            absorb = false(1,nstate);
            nextstate = obj.A*state + obj.B*action;
            reward = zeros(obj.dreward,nstate);
            
            for i = 1 : obj.dreward
                reward(i,:) = -sum(bsxfun(@times, state'*obj.Q(:,:,i), state'), 2)' ...
                    -sum(bsxfun(@times, action'*obj.R(:,:,i), action'), 2)';
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

        function fig = plotfront(obj, front, varargin)
            fig = plotfront@MOMDP(front, varargin{:});
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