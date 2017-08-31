classdef PolicyTabular < Policy
% POLICYTABULAR The probability of drawing action A in state S is stored in
% a matrix P of size [DS X DA], where DS is the number of all states and DA
% is the number of all actions.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        P
        state_list % Matrix containing all possible states (row-wise)
        action_list % Vector containing all possible actions
    end
    
    methods
        
        %% Constructor
        function obj = PolicyTabular(state_list, action_list)
            assert(isvector(action_list), ...
                'Action list is not a vector.')

            obj.state_list = state_list;
            obj.action_list = action_list;
            obj.daction = 1;
            obj.P = 0.5 * ones(size(state_list,1), length(action_list));
        end
        
        %% Distribution
        function Actions = drawAction(obj, States)
            [~,idx] = ismember(States',obj.state_list,'rows');
            Actions = mymnrnd(obj.P(idx,:)',size(States,2));
        end
        
        function S = entropy(obj, varargin)
            idx = obj.P~=0;
            S = -sum(sum(obj.P(idx).*log2(obj.P(idx))));
        end

        %% Update
        function obj = update(obj, P)
            obj.P = P;
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            [~,idx] = max(obj.P,[],2);
            idx = sub2ind([size(obj.state_list,1), ...
                length(obj.action_list)], ...
                1:size(obj.state_list,1),idx');
            obj.P(:) = 0;
            obj.P(idx) = 1;
        end
        
    end
    
end
