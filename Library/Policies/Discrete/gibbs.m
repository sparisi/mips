classdef gibbs < policy_discrete
% GIBBS Gibbs (softmax) distribution with preferences on all but last 
% action. The temperature is fixed.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        epsilon % temperature (low value -> deterministic policy)
    end
    
    methods
        
        function obj = gibbs(basis, theta, action_list)
            assert(isvector(action_list))
            assert(basis()*(length(action_list)-1) == length(theta))

            obj.basis = basis;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.epsilon = 1;
            obj.dim_explore = 0;
            obj.dim = 1;
        end
  
        function [prob_list, qfun] = distribution(obj, States)
            nstates = size(States,2);
            temperature = obj.epsilon;
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
                num(i,:) = exp(loc_Q/temperature);
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
            
            temperature = obj.epsilon;
            nactions = length(obj.action_list) - 1;
            exp_term = zeros(nactions,1);
            dphi = feval(obj.basis);
            phi_term = zeros(dphi*nactions,nactions);
            phi = feval(obj.basis,state);
            for i = 1 : nactions
                loc_phi = zeros(dphi*nactions,1);
                loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
                loc_Q = obj.theta'*loc_phi;
                exp_term(i) = exp(loc_Q/temperature);
                phi_term(:,i) = loc_phi/temperature;
            end
            
            idx = isinf(exp_term);
            if max(idx)
                exp_term(idx) = 1;
                exp_term(~idx) = 0;
            end
            
            num = phi_term * exp_term;
            den = sum(exp_term);

            if action == obj.action_list(end)
                dlpdt = - num ./ (1 + den);
            else
                loc_phi = zeros(dphi*nactions,1);
                i = find(obj.action_list == action);
                loc_phi((i-1)*dphi+1:(i-1)*dphi+dphi) = phi;
                dlpdt = loc_phi/temperature - num ./ (1 + den);
            end
        end
        
        % Basis function depending on the action
        function phiA = basisA(obj, States, Actions)
            dphi = feval(obj.basis);
            nactions = length(obj.action_list);
            if nargin == 1
                phiA = dphi * (nactions - 1);
                return
            end
            
            nstates = size(States,2);
            [found,~] = (ismember(Actions,obj.action_list));
            assert(min(found) == 1);
            assert(isrow(Actions))
            assert(length(Actions) == nstates)

            phi = obj.basis(States); % Compute phi(s)
            start_idx = (Actions-1)*dphi+1 + [0:nstates-1]*dphi*nactions; % Column start linear indices
            all_idx = bsxfun(@plus,start_idx,[0:dphi-1]'); % All linear indices
            phiA = zeros(dphi*nactions,nstates); % Initialize output array with zeros
            phiA(all_idx) = phi; % Insert values from phi into output array
            phiA(end-dphi+1:end,:) = []; % Last action does not have any explicit preference
        end
        
        function obj = makeDeterministic(obj)
            obj.epsilon = 1e-8;
        end
        
    end
    
end
