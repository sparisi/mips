classdef gibbs < policy_discrete
% GIBBS Gibbs (softmax) distribution with preferences on all but last 
% action. The temperature is fixed.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        action_list;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        inverse_temperature;
    end
    
    methods
        
        function obj = gibbs(basis, theta, action_list)
            assert(isvector(action_list))
            assert(basis()*(length(action_list)-1) == length(theta))

            obj.basis = basis;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.inverse_temperature = 1;
            obj.dim_explore = 0;
            obj.dim = 1;
        end
  
        function [prob_list, qfun] = distribution(obj, States)
            nstates = size(States,2);
            it = obj.inverse_temperature;
            nactions = length(obj.action_list);
            num = zeros(nactions, nstates);
            qfun = zeros(nactions, nstates);
            den = zeros(1, nstates);
            dphi = feval(obj.basis);
            phi = feval(obj.basis,States);
            for i = 1 : nactions - 1
                loc_phi = zeros(dphi*(nactions-1),nstates);
                loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi,:) = phi;
                loc_Q = obj.theta'*loc_phi;
                num(i,:) = exp(it*loc_Q);
                qfun(i,:) = loc_Q;
                den = den + num(i,:);
            end
            den = den + 1;
            num(end,:) = 1;
            qfun(end,:) = 0;
            
            prob_list = bsxfun(@times,num,1./den);
            % A NaN can occur if the exp was Inf
            prob_list(isnan(prob_list)) = 1;
            % Ensure that the sum is 1
            prob_list = bsxfun(@times,prob_list,1./sum(prob_list));
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlpdt = size(obj.theta,1);
                return
            end
            assert(size(state,2) == 1);
            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            it = obj.inverse_temperature;
            nactions = length(obj.action_list) - 1;
            exp_term = zeros(nactions,1);
            dphi = feval(obj.basis);
            phi_term = zeros(dphi*nactions,nactions);
            phi = feval(obj.basis,state);
            for i = 1 : nactions
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
            
            if action == obj.action_list(end)
                dlpdt = - num ./ (1 + den);
            else
                dlpdt = it*loc_phi - num ./ (1 + den);
            end
        end
        
        % Basis function depending from the action
        function Aphi = Abasis(obj, state, action)
            assert(size(state,2) == 1)
            i = find(obj.action_list == action);
            assert(length(i) == 1);

            dphi = feval(obj.basis);
            nactions = length(obj.action_list) - 1;
            if nargin == 1
                Aphi = dphi * nactions;
                return
            end
            phi = obj.basis(state);
            Aphi = zeros(dphi*nactions,1);
            Aphi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
        end
        
        function obj = makeDeterministic(obj)
            obj.inverse_temperature = 1e8;
        end
        
    end
    
end
