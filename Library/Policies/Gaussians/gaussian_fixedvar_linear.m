classdef gaussian_fixedvar_linear < policy_gaussian
% GAUSSIAN_FIXEDVAR_LINEAR Gaussian distribution with linear mean and fixed 
% covariance: N(A*phi,S).
% Parameters: mean A.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        Sigma;
    end
    
    methods
        
        function obj = gaussian_fixedvar_linear(basis, dim, initA, Sigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(initA,2))
            assert(dim == size(initA,1))
            assert(size(Sigma,1) == dim)
            assert(size(Sigma,2) == dim)
            [~, p] = chol(Sigma);
            assert(p == 0)

            obj.theta = initA(:);
            obj.basis = basis;
            obj.dim = dim;
            obj.Sigma = Sigma;
            obj.dim_explore = 0;
        end
        
        function params = getParams(obj, state)
            A = vec2mat(obj.theta,obj.dim);
            params.A = A;
            params.a = 0;
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
            params = obj.getParams;
            A = params.A;
            dlpdt = (obj.Sigma) \ (action - A*phi) * phi';
            dlpdt = dlpdt(:);
        end
        
        %%% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            if nargin == 1
                hlpdt = size(obj.theta,1);
                return
            end
            
            phi = feval(obj.basis, state);
            phimat = kron(phi',eye(obj.dim));
            hlpdt = -phimat' / obj.Sigma * phimat;
        end
        
        function obj = makeDeterministic(obj)
            obj.Sigma = 1e-8 * eye(size(obj.Sigma));
        end
        
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0) % weights cannot be negative
            D = diag(weights);
            A = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Action;
            A = A';
            obj.theta = A(:);
        end
        
    end
    
end
