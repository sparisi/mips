classdef gaussian_fixedvar_constant < policy_gaussian
% GAUSSIAN_FIXEDVAR_CONSTANT Gaussian distribution with constant mean and 
% fixed covariance: N(mu,S).
% Parameters: mean mu.
    
    properties(GetAccess = 'public', SetAccess = 'public')
        Sigma;
    end
    
    methods
        
        function obj = gaussian_fixedvar_constant(dim, initMean, Sigma)
            assert(isscalar(dim))
            assert(size(initMean,1) == dim)
            assert(size(initMean,2) == 1)
            assert(size(Sigma,1) == dim)
            assert(size(Sigma,2) == dim)
            [~, p] = chol(Sigma);
            assert(p == 0)
            
            obj.theta = initMean;
            obj.dim = dim;
            obj.dim_explore = 0;
            obj.Sigma = Sigma;
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            params.mu = mu;
            params.Sigma = obj.Sigma;
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = length(obj.theta);
                return
            end
            
            params = obj.getParams;
            mu = params.mu;
            dlogpdt = obj.Sigma \ (action - mu);
        end
        
        %%% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            if nargin == 1
                hlpdt = size(obj.theta,1);
                return
            end
            
            hlpdt = -inv(obj.Sigma);
        end
        
        function obj = makeDeterministic(obj)
            obj.Sigma = 1e-8 * eye(size(obj.Sigma));
        end
        
        function obj = weightedMLUpdate(obj, weights, Actions)
            assert(min(weights)>=0) % weights cannot be negative
            mu = Actions * weights / sum(weights);
            obj.theta = mu;
        end        
        
    end
    
end
