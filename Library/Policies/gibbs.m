%%% Gibbs (soft-max) policy with preferences on all but last action.
%%% The temperature is fixed.
classdef gibbs < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        action_list;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        inverse_temperature;
    end
    
    methods
        
        function obj = gibbs(bfs, theta, action_list)
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
            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            num = zeros(nactions, 1);
            den = 0;
            for i = 1 : nactions - 1
                loc_phi = feval(obj.basis, state, obj.action_list(i));
                loc_Q = obj.theta'*loc_phi;
                num(i) = exp(it*loc_Q);
                den = den + num(i);
            end
            den = den + 1;
            num(end) = 1;
            
            prob_list = num / den;
            % A NaN can occur if the exp was Inf
            prob_list(isnan(prob_list)) = 1;
            % Ensure that the sum is 1
            prob_list = prob_list / sum(prob_list);
            
            % Get action probability
            probability = prob_list(idx);
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            num = zeros(nactions, 1);
            den = 0;
            for i = 1 : nactions - 1
                loc_phi = feval(obj.basis, state, obj.action_list(i));
                loc_Q = obj.theta'*loc_phi;
                num(i) = exp(it*loc_Q);
                den = den + num(i);
            end
            den = den + 1;
            num(end) = 1;
            
            prob_list = num / den;
            % A NaN can occur if the exp was Inf
            prob_list(isnan(prob_list)) = 1;
            % Ensure that the sum is 1
            prob_list = prob_list / sum(prob_list);
            [~, action] = find(mnrnd(1, prob_list));
        end
        
        function S = entropy(obj, state)
            assert(size(state,2) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            num = zeros(nactions, 1);
            den = 0;
            for i = 1 : nactions - 1
                loc_phi = feval(obj.basis, state, obj.action_list(i));
                loc_Q = obj.theta'*loc_phi;
                num(i) = exp(it*loc_Q);
                den = den + num(i);
            end
            den = den + 1;
            num(end) = 1;
            
            prob_list = num / den;
            % A NaN can occur if the exp was Inf
            prob_list(isnan(prob_list)) = 1;
            % Ensure that the sum is 1
            prob_list = prob_list / sum(prob_list);

            S = 0;
            for i = 1 : nactions
                % Usual checks for the entropy
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
            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list) - 1;
            exp_term = zeros(nactions,1);
            phi_term = zeros(feval(obj.basis),nactions);
            for i = 1 : nactions
                loc_phi = feval(obj.basis, state, obj.action_list(i));
                loc_Q = obj.theta'*loc_phi;
                exp_term(i) = exp(it*loc_Q);
                phi_term(:,i) = it*loc_phi;
            end
            
            idx = isinf(exp_term);
            if max(idx)
                exp_term(idx) = 1;
                exp_term(~idx) = 0;
            end
            
            num = phi_term * exp_term;
            den = sum(exp_term);

            phi = feval(obj.basis, state, action);
            
            if action == obj.action_list(end)
                dlpdt = - num ./ (1 + den);
            else
                dlpdt = it*phi - num ./ (1 + den);
            end
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.inverse_temperature = 1e8;
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                % Return the dimension of the vector of basis functions
                phi = feval(obj.basis) / (length(obj.action_list) - 1);
                return
            end
            phi = feval(obj.basis, state);
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
                areEq = bitand(areEq, areEqTemp);
            else
                return;
            end
        end
        
    end
    
end
