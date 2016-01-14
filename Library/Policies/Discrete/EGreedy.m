classdef EGreedy < PolicyDiscrete
% EGREEDY Epsilon-greedy distribution with preferences on all but last 
% action.
    
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
            assert((basis()+1)*(length(action_list)-1) == length(theta), ...
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
            [~, idx] = max(Q,[],1);
            nactions = length(obj.action_list);
            nstates = size(States,2);
            linearIdx = [0:nstates-1]*nactions + idx;
            prob_list = obj.epsilon / nactions * ones(nactions,nstates);
            prob_list(linearIdx) = prob_list(linearIdx) + 1 - obj.epsilon;
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
