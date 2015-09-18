classdef gibbs_allpref < policy
% GIBBS_ALLPREF Gibbs (soft-max) distribution with preferences on all 
% actions. The temperature is fixed.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        action_list;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        inverse_temperature;
    end
    
    methods
        
        function obj = gibbs_allpref(basis, theta, action_list)
            % Class constructor
            obj.basis = basis;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.inverse_temperature = 1;
            obj.dim_explore = 0;
            obj.dim = 1;
        end
        
        function prob_list = distribution(obj, state)
            assert(size(state,2) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            num = zeros(nactions, 1);
            den = 0;
            dphi = feval(obj.basis);
            phi = feval(obj.basis,state);
            for i = 1 : nactions
                loc_phi = zeros(dphi*nactions,1);
                loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
                loc_Q = obj.theta'*loc_phi;
                num(i) = exp(it*loc_Q);
                den = den + num(i);
            end
            
            prob_list = num / den;
            % A NaN can occur if the exp was Inf
            prob_list(isnan(prob_list)) = 1;
            % Ensure that the sum is 1
            prob_list = prob_list / sum(prob_list);
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);

            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            % Get action probability
            prob_list = obj.distribution(state);
            probability = prob_list(idx);
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            
            prob_list = obj.distribution(state);
            [~, action] = find(mnrnd(1, prob_list));
        end
        
        function S = entropy(obj, state)
            assert(size(state,2) == 1);
            
            nactions = length(obj.action_list);
            prob_list = obj.distribution(state);

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
            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            exp_term = zeros(nactions,1);
            dphi = feval(obj.basis);
            phi_term = zeros(dphi*nactions,nactions);
            phi = feval(obj.basis,state);
            for i = 1 : nactions;
                loc_phi = zeros(dphi*nactions,1);
                loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
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
            
            loc_phi = zeros(dphi*nactions,1);
            i = find(obj.action_list == action);
            loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
            dlpdt = it*loc_phi - num ./ den;
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
                areEq = bitand(areEq, areEqTemp);
            else
                return;
            end
        end
        
        % Basis function depending from the action
        function Aphi = Abasis(obj, state, action)
            dphi = feval(obj.basis);
            nactions = length(obj.action_list);
            if nargin == 1
                Aphi = dphi * nactions;
                return
            end
            phi = obj.basis(state);
            Aphi = zeros(dphi*nactions,1);
            i = find(obj.action_list == action);
            Aphi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
        end
        
    end
    
end
