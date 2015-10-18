classdef gaussian_logistic_constant < policy_gaussian
% GAUSSIAN_LOGISTIC_CONSTANT Gaussian distribution with constant mean and 
% logistic covariance: N(mu,S).
% Parameters: mean mu and logistic weights w, with S = tau/(1+exp(-w)).
    
    properties(GetAccess = 'public', SetAccess = 'public')
        tau;
    end
    
    methods
        
        function obj = ...
                gaussian_logistic_constant(dim, initMean, initW, maxVar)
            assert(isscalar(dim))
            assert(size(initMean,1) == dim)
            assert(size(initMean,2) == 1)
            assert(size(initW,1) == dim)
            assert(size(initW,2) == 1)
            assert(size(maxVar,1) == size(initW,1));
            assert(size(maxVar,2) == size(initW,2));

            obj.theta = [initMean; initW];
            obj.dim = dim;
            obj.tau = maxVar;
            obj.dim_explore = length(initW);
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            logv = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            Sigma = diag(logv);
            params.mu = mu;
            params.Sigma = Sigma;
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            
            logv = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            mu = obj.theta(1:obj.dim);
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = (action - mu) ./ logv;
            dlpdt(1:end-obj.dim) = dmu(:);
            
            for i = 1 : obj.dim
                wi = obj.theta(end-obj.dim+i);
                ds1 = -0.5 * exp(-wi) / (1 + exp(-wi));
                ds2 = 0.5 * exp(-wi) / obj.tau(i) * (action(i) - mu(i))^2;
                dlpdt(end-obj.dim+i) = ds1 + ds2;
            end
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights)>=0) % weights cannot be negative
            weights = weights / sum(weights);
            mu = Action * weights;
            logv = zeros(obj.dim,1);
            
            for j = 1 : obj.dim
                tmp = 0;
                for k = 1 : size(Action,2)
                    tmp = tmp + (weights(k) * (Action(j,k) - mu(j)).^2);
                end
                logv(j) = -log( obj.tau(j) / tmp - 1 );
            end
                
            obj.theta = [mu; logv];
        end
        
    end
    
end
