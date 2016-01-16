classdef Gibbs < PolicyDiscrete
% GIBBS Gibbs (softmax) distribution with preferences on all but last 
% action. The temperature is fixed.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        epsilon % temperature (low value -> deterministic policy)
    end
    
    methods
        
        %% Constructor
        function obj = Gibbs(basis, theta, action_list)
            assert(isvector(action_list), ...
                'Action list is not a vector.')
            assert((basis()+1)*(length(action_list)-1) == length(theta), ...
                'Wrong number of initial parameters.')

            obj.basis = basis;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.epsilon = 1;
            obj.daction = 1;
            obj.dparams = length(obj.theta);
        end
        
        %% Distribution
        function prob_list = distribution(obj, States)
            Q = obj.qFunction(States);
            Q = bsxfun(@minus, Q, max(Q,[],1)); % Numerical trick to avoid Inf and NaN (the distribution does not change)
            exp_term = exp(Q / obj.epsilon);
            prob_list = bsxfun(@times, exp_term, 1./sum(exp_term,1));
            prob_list = bsxfun(@times, prob_list, 1./sum(prob_list)); % Ensure that the sum is 1
        end

        %% Q-function
        function Q = qFunction(obj, States, Actions)
        % If no actions are provided, the function returns the Q-function
        % for all possible actions.
            nstates = size(States,2);
            lactions = length(obj.action_list);
            phi = obj.basis1(States);
            dphi = size(phi,1);

            Q = [reshape(obj.theta,dphi,lactions-1)'*phi;
                zeros(1,nstates)]; % last action has 0 weights

            if nargin == 3
                idx = ([1:nstates]-1)*dphi+Actions;
                Q = Q(idx); % linear indexing
            end
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, States, Actions)
            assert(size(States,2) == size(Actions,2), ...
                'The number of states and actions must be the same.');
            found = (ismember(Actions,obj.action_list));
            assert(min(found) == 1, 'Unknown action.');

            phi = obj.basis1(States);
            dphi = size(phi,1);
            prob_list = obj.distribution(States);

            dlpdt = -mtimescolumn(phi, prob_list(1:end-1,:)) / obj.epsilon;
            for i = 1 : length(obj.action_list) - 1
                idx1 = (i-1)*dphi + 1 : (i-1)*dphi + dphi;
                idx2 = Actions == i;
                dlpdt(idx1,idx2) = dlpdt(idx1,idx2) + phi(:,idx2) / obj.epsilon;
            end
        end
        
        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.epsilon = 1e-8;
        end
        
        function obj = randomize(obj, factor)
            obj.theta = obj.theta ./ factor;
        end
        
    end
    
end
