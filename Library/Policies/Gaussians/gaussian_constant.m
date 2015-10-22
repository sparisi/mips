classdef gaussian_constant < policy_gaussian
% GAUSSIAN_CONSTANT Gaussian distribution with constant mean and
% covariance: N(mu,S).
% Parameters: mean mu and covariance S.
    
    methods
        
        function obj = gaussian_constant(dim, initMean, initSigma)
            assert(isscalar(dim))
            assert(size(initMean,1) == dim)
            assert(size(initMean,2) == 1)
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [~, p] = chol(initSigma);
            assert(p == 0)
            
            obj.theta = [initMean; initSigma(:)];
            obj.dim = dim;
            obj.dim_explore = length(initSigma(:));
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            Sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            params.mu = mu;
            params.Sigma = Sigma;
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = length(obj.theta);
                return
            end
            
            params = obj.getParams;
            Sigma = params.Sigma;
            mu = params.mu;
            
            dlogpdt_mu = Sigma \ (action - mu);
            invSigma = inv(Sigma)';
            ds1 = -0.5 * invSigma;
            ds2 =  0.5 * invSigma * (action - mu) * (action - mu)' * invSigma;
            dlogpdt_sigma = ds1 + ds2;
            dlogpdt = [dlogpdt_mu; dlogpdt_sigma(:)];
        end
        
        function obj = weightedMLUpdate(obj, weights, Actions)
            assert(min(weights)>=0) % weights cannot be negative
            mu = Actions * weights / sum(weights);
            Sigma = zeros(obj.dim);
            for k = 1 : size(Actions,2)
                Sigma = Sigma + (weights(k) * (Actions(:,k) - mu) * (Actions(:,k) - mu)');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            obj.theta = [mu; Sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            % Additional check for positivity
            params = obj.getParams;
            Sigma = params.Sigma;
            [~, check] = chol(Sigma);
            if check ~= 0
                Sigma = nearestSPD(Sigma);
            end
            obj.theta(obj.dim+1:end) = Sigma(:);
        end
        
    end
    
end
