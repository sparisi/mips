classdef EGreedy < PolicyDiscrete
% EGREEDY Epsilon-greedy distribution with preferences on all actions.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        epsilon
    end
    
    methods
        
        %% Constructor
        function obj = EGreedy(basis, theta, action_list, epsilon)
            assert(isvector(action_list), ...
                'Action list is not a vector.')
            assert((basis()+1)*length(action_list) == length(theta), ...
                'Wrong number of initial parameters.')

            obj.basis = basis;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.epsilon = epsilon;
            obj.daction = 1;
            obj.dparams = length(obj.theta);
        end
        
        %% Distribution
        function prob_list = distribution(obj, States)
            Q = obj.qFunction(States);
            maxval = max(Q,[],1);
            idx = bsxfun(@ismember,Q,maxval);
            nactions = length(obj.action_list);
            nstates = size(States,2);
            prob_list = obj.epsilon / nactions * ones(nactions,nstates);
            remainder = bsxfun(@times, (1 - obj.epsilon) * ones(size(prob_list)), 1 ./ sum(idx,1));
            prob_list(idx) = prob_list(idx) + remainder(idx);
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

            Q = reshape(obj.theta,dphi,lactions)'*phi;

            if nargin == 3
                idx = (0:nstates-1)*lactions+Actions;
                Q = Q(idx); % linear indexing
            end
        end
        
        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.epsilon = 0;
        end
        
        function obj = randomize(obj, factor)
            obj.theta = obj.theta ./ factor;
        end
        
    end
    
end
