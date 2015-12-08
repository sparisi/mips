classdef gaussian_fixedvar_linear_diagmean < policy_gaussian
% GAUSSIAN_FIXEDVAR_LINEAR_DIAGMEAN Gaussian distribution with linear 
% diagonal mean and fixed covariance: N(A*phi,S), A diagonal matrix.
% Parameters: mean A.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        Sigma;
    end
    
    methods
        
        function obj = gaussian_fixedvar_linear_diagmean(basis, dim, initA, Sigma)
            assert(isscalar(dim))
            assert(feval(basis) == dim)
            assert(size(initA,2) == 1)
            assert(size(initA,1) == dim)
            assert(size(Sigma,1) == dim)
            assert(size(Sigma,2) == dim)
            [~, p] = chol(Sigma);
            assert(p == 0);

            obj.theta = initA;
            obj.basis = basis;
            obj.dim = dim;
            obj.Sigma = Sigma;
            obj.dim_explore = 0;
        end
        
        function params = getParams(obj, state)
            A = diag(obj.theta);
            params.A = A;
            params.Sigma = obj.Sigma;
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
            phi = feval(obj.basis, state);
            A = diag(obj.theta);
            dlpdt = (obj.Sigma) \ (action - A * phi) * phi';
            dlpdt = diag(dlpdt);
        end
        
        %%% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            if nargin == 1
                hlpdt = size(obj.theta,1);
                return
            end
            
            phi = feval(obj.basis, state);
            hlpdt = -phi * phi' .* inv(obj.Sigma);
        end
        
        function obj = makeDeterministic(obj)
            obj.Sigma = 1e-8 * eye(size(obj.Sigma));
        end
        
    end
    
end
