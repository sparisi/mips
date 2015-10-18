classdef gaussian_linear < policy_gaussian
% GAUSSIAN_LINEAR Gaussian distribution with linear mean and constant 
% covariance: N(A*phi,S).
% Parameters: mean A and covariance S.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    methods
        
        function obj = gaussian_linear(basis, dim, initA, initSigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(initA,2))
            assert(dim == size(initA,1))
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [~, p] = chol(initSigma);
            assert(p == 0)

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [initA(:); initSigma(:)];
            obj.dim_explore = length(initSigma(:));
        end
        
        function params = getParams(obj, States)
            n = obj.dim*feval(obj.basis);
            A = vec2mat(obj.theta(1:n),obj.dim);
            Sigma = vec2mat(obj.theta(n+1:end),obj.dim);
            params.A = A;
            params.a = 0;
            params.Sigma = Sigma;
            if nargin == 2
                phi = feval(obj.basis, States);
                params.mu = A*phi;
            end
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            
            phi = feval(obj.basis, state);
            params = obj.getParams;
            Sigma = params.Sigma;
            A = params.A;
            mu = A*phi;
            
            dlogpdt_A = Sigma \ (action - mu) * phi';
            ds1 = -0.5 * eye(obj.dim) / Sigma;
            ds2 = 0.5 * Sigma \ (action - mu) * (action - mu)' / Sigma;
            dlogpdt_sigma = ds1 + ds2;
            dlogpdt = [dlogpdt_A(:); dlogpdt_sigma(:)];
        end
        
        function obj = weightedMLUpdate(obj, weights, Actions, Phi)
            assert(min(weights)>=0) % weights cannot be negative
            Sigma = zeros(obj.dim);
            D = diag(weights);
            N = size(Actions,1);
            A = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Actions;
            A = A';
            for k = 1 : N
                Sigma = Sigma + (weights(k) * (Actions(k,:)' - A*Phi(k,:)') * (Actions(k,:)' - A*Phi(k,:)')');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            obj.theta = [A(:); Sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            n = obj.dim*feval(obj.basis);
            sigma = vec2mat(obj.theta(n+1:end),obj.dim);
            obj.theta(n+1:end) = nearestSPD(sigma); % Ensure positivity
        end
        
    end
    
end
