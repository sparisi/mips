classdef gaussian_logistic_linear < policy_gaussian
% GAUSSIAN_LOGISTIC_LINEAR Gaussian distribution with linear mean and 
% logistic covariance: N(A*phi,S).
% Parameters: mean A and logistic weights w, with S = tau/(1+exp(-w)).
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end

    properties(GetAccess = 'public', SetAccess = 'public')
        tau;
    end
    
    methods
        
        function obj = ...
                gaussian_logistic_linear(basis, dim, initMean, initW, maxVar)
            assert(isscalar(dim))
            assert(feval(basis) == size(initMean,2))
            assert(dim == size(initMean,1))
            assert(size(initW,1) == dim)
            assert(size(initW,2) == 1)
            assert(size(maxVar,1) == size(initW,1));
            assert(size(maxVar,2) == size(initW,2));

            obj.theta = [initMean(:); initW];
            obj.basis = basis;
            obj.dim = dim;
            obj.tau = maxVar;
            obj.dim_explore = length(initW);
        end
        
        function params = getParams(obj, state)
            logv = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            Sigma = diag(logv);
            A = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            params.A = A;
            params.a = 0;
            params.Sigma = Sigma;
            if nargin == 2
                phi = feval(obj.basis, state);
                params.mu = A*phi;
            end
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlpdt = size(obj.theta,1);
                return
            end
            
            logv = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            invSigma = diag(1./logv);
            phi = feval(obj.basis, state);
            A = vec2mat(obj.theta(1:end-obj.dim),obj.dim);
            mu = A * phi;
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = invSigma * (action - mu) * phi';
            dlpdt(1:end-obj.dim) = dmu(:);
            
            for i = 1 : obj.dim
                wi = obj.theta(end-obj.dim+i);
                ds1 = -0.5 * exp(-wi) / (1 + exp(-wi));
                ds2 = 0.5 * exp(-wi) / obj.tau(i) * (action(i) - mu(i))^2;
                dlpdt(end-obj.dim+i) = ds1 + ds2;
            end
        end
        
    end
    
end
