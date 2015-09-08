classdef gaussian_constant < policy
% GAUSSIAN_CONSTANT Gaussian distribution with constant mean and
% covariance.
% Parameters: mean K and covariance S.
    
    methods
        
        function obj = gaussian_constant(dim, init_mean, init_sigma)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)
            
            obj.theta = [init_mean; init_sigma(:)];
            obj.dim = dim;
            obj.dim_explore = length(init_sigma(:));
        end
        
        function action = drawAction(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            action = mvnrnd(mu,sigma)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj)
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma) );
        end
        
        function probability = evaluate(obj, action)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            probability = mvnpdf(action, mu, sigma);
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = length(obj.theta);
                return
            end
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            
            dlogpdt_mu = sigma \ (action - mu);
            invsigma = inv(sigma)';
            A = -0.5 * invsigma;
            B =  0.5 * invsigma * (action - mu) * (action - mu)' * invsigma;
            dlogpdt_sigma = A + B;
            
            dlogpdt = [dlogpdt_mu; dlogpdt_sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            obj.theta(obj.dim+1:end) = nearestSPD(sigma);
        end
        
        function obj = makeDeterministic(obj)
            obj.theta(obj.dim+1:end) = 1e-8;
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = vec2mat(obj.theta(obj.dim+1:end),obj.dim);
            
            params.mu = mu;
            params.Sigma = sigma;
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights)>=0) % weights cannot be negative
            mu = Action * weights / sum(weights);
            sigma = zeros(size(obj.dim));
            for k = 1 : size(Action,2)
                sigma = sigma + (weights(k) * (Action(:,k) - mu) * (Action(:,k) - mu)');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            sigma = sigma / Z;
            sigma = nearestSPD(sigma);
            obj.theta = [mu; sigma(:)];
        end
        
        function obj = randomize(obj, factor)
            obj.theta(obj.dim+1:end) = obj.theta(obj.dim+1:end) .* factor;
        end
        
        function plot(obj)
            params = obj.getParams;
            mu = params.mu;
            Sigma = params.Sigma;
            figure; hold all
            xlabel 'x_i'
            ylabel 'Policy density'
            x = max(abs(mu)) + 2*max(abs(Sigma(:)));
            range = -x: 0.1 : x;
            for i = 1 : length(mu)
                norm = normpdf(range, mu(i), Sigma(i,i));
                plot(range, norm)
            end
        end
        
    end
    
end
