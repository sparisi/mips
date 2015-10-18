classdef gaussian_linear_full < policy_gaussian
% GAUSSIAN_LINEAR_FULL Gaussian distribution with linear mean (with offset) 
% and constant covariance: N(a+A*phi,S).
% Parameters: offset a, mean A and covariance S.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    methods
        
        function obj = ...
                gaussian_linear_full(basis, dim, initMean, initA, initSigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(initA,2))
            assert(dim == size(initA,1))
            assert(dim == size(initMean,1))
            assert(size(initMean,2) == 1)
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [~, p] = chol(initSigma);
            assert(p == 0)

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [initMean; initA(:); initSigma(:)];
            obj.dim_explore = length(initSigma(:));
        end
        
        function params = getParams(obj, state)
            n = obj.dim*feval(obj.basis)+obj.dim;
            a = obj.theta(1:obj.dim);
            A = vec2mat(obj.theta(obj.dim+1:n),obj.dim);
            Sigma = vec2mat(obj.theta(n+1:end),obj.dim);
            params.a = a;
            params.A = A;
            params.Sigma = Sigma;
            if nargin == 2
                phi = feval(obj.basis, state);
                params.mu = repmat(a,1,size(state,2))+A*phi;
            end
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis,state);
            params = obj.getParams(state);
            a = params.a;
            A = params.A;
            Sigma = params.Sigma;
            mu = params.mu;
            
            dlogpdt_offset = Sigma \ a;
            dlogpdt_A = Sigma \ (action - A*phi) * phi';

            ds1 = -.5 * inv(Sigma);
            ds2 = .5 * Sigma \ (action - mu) * (action - mu)' / Sigma;
            dlogpdt_sigma = ds1 + ds2;
            dlogpdt = [dlogpdt_offset; dlogpdt_A(:); dlogpdt_sigma(:)];
        end
        
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0) % weights cannot be negative
            N = size(Action,1);
            D = diag(weights);
            S = [ones(N,1), Phi];
            W = (S' * D * S + 1e-8 * eye(size(S,2))) \ S' * D * Action;
            a = W(1,:)';
            A = W(2:end,:)';
            Sigma = zeros(obj.dim);
            for k = 1 : N
                mu = a + A * Phi(k,:)';
                Sigma = Sigma + (weights(k) * (Action(k,:)' - mu) * (Action(k,:)' - mu)');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            obj.theta = [a; A(:); Sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            n = obj.dim*feval(obj.basis)+obj.dim;
            sigma = vec2mat(obj.theta(n+1:end),obj.dim);
            obj.theta(n+1:end) = nearestSPD(sigma); % Ensure positivity
        end
        
    end
    
end
