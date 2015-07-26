%%% Gibbs (soft-max) policy with preferences on all action.
%%% The temperature is fixed.
classdef gibbs_allpref < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        action_list;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        inverse_temperature;
    end
    
    methods
        
        function obj = gibbs_allpref(bfs, theta, action_list)
            % Class constructor
            obj.basis = bfs;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.inverse_temperature = 1;
            obj.dim_explore = 0;
            obj.dim = 1;
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);

            IT = obj.inverse_temperature;
            
            % Assert that the action is one of the discrete known actions
            I = find(obj.action_list == action);
            assert(length(I) == 1);
            
            % Evaluate the bfs of the current state and action
            phi = feval(obj.basis, state, action);
            lin_prod = obj.theta'*phi;
            
            % Compute the sum of all the preferences
            sumexp = 0;
            for i = 1 : length(obj.action_list)
                act = obj.action_list(i);
                local_phi = feval(obj.basis, state, act);
                sumexp = sumexp + min(exp(IT*obj.theta'*local_phi),1e200);
            end
            
            % Compute action probability
            probability = min(exp(IT*lin_prod),1e200) / sumexp;
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            
            IT = obj.inverse_temperature;
            
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            
            % Compute the sum of all the preferences
            sumexp = 0;
            
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                exp_term = min(exp(IT*obj.theta'*loc_phi),1e200);
                prob_list(i) = exp_term;
                sumexp = sumexp + exp_term;
            end
            prob_list = prob_list / sumexp;
            prob_list(isnan(prob_list)) = 1;
            prob_list(isinf(prob_list)) = 1;
            prob_list = prob_list / sum(prob_list);
            [~, action] = find(mnrnd(1, prob_list));
        end
        
        function S = entropy(obj, state)
            assert(size(state,2) == 1);
            
            IT = obj.inverse_temperature;
            
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            
            sumexp = 0;
            
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                exp_term = min(exp(IT*obj.theta'*loc_phi),1e200);
                prob_list(i) = exp_term;
                sumexp = sumexp + exp_term;
            end
            prob_list = prob_list / sumexp;
            S = 0;
            for i = 1 : nactions
                if ~(isinf(prob_list(i)) || isnan(prob_list(i)) || prob_list(i) == 0)
                    S = S + (-prob_list(i)*log2(prob_list(i)));
                end
            end
            S = S / log2(nactions);
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlpdt = size(obj.theta,1);
                return
            end
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
            
            IT = obj.inverse_temperature;
            
            % Assert that the action is one of the discrete known actions
            I = find(obj.action_list == action);
            assert(length(I) == 1);
            
            % Compute the sum of all the preferences
            sumexp = 0;
            sumpref = 0;
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                exp_term = min(exp(IT*obj.theta'*loc_phi),1e200);
                prob_list(i) = exp_term;
                sumexp = sumexp + exp_term;
                sumpref = sumpref + IT*loc_phi*prob_list(i);
            end
            sumpref = sumpref / sumexp;
            
            loc_phi = feval(obj.basis, state, action);
            dlpdt = IT*loc_phi - sumpref;
        end
        
        function obj = makeDeterministic(obj)
            obj.inverse_temperature = 1e8;
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = randomize(obj, factor)
            obj.theta = obj.theta ./ factor;
        end
        
        function areEq = eq(obj1, obj2)
            areEq = eq@policy(obj1,obj2);
            if max(areEq)
                areEqTemp = bsxfun( @and, [obj1(:).inverse_temperature], [obj2(:).inverse_temperature] );
                if size(areEq,1) ~= size(areEqTemp,1)
                    areEqTemp = areEqTemp';
                end
                areEq = bitand( areEq, areEqTemp);
            else
                return;
            end
        end
        
    end
    
end
