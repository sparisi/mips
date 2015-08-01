%%% Gaussian with constant mean and diagonal covariance: N(K,S).
%%% Params: mean and diagonal std (S = diag(s)^2).
classdef gaussian_diag_constant < policy
    
    methods
        
        function obj = gaussian_diag_constant(dim, init_mean, init_sigma)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == 1)
            
            obj.theta = [init_mean; init_sigma];
            obj.dim = dim;
            obj.dim_explore = length(init_sigma);
        end
        
        function action = drawAction(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = diag(obj.theta(obj.dim+1:end));
            action = mvnrnd(mu,sigma^2)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj)
            sigma = diag(obj.theta(obj.dim+1:end));
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma^2) );
        end
        
        function probability = evaluate(obj, action)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = diag(obj.theta(obj.dim+1:end));
            probability = mvnpdf(action, mu, sigma^2);
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = length(obj.theta);
                return
            end
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = obj.theta(obj.dim+1:end);
            
            dlogpdt_mu = sigma.^-2 .* (action - mu);
            dlogpdt_sigma = -sigma.^-1 + (action - mu).^2 ./ sigma.^3;
            
            dlogpdt = [dlogpdt_mu; dlogpdt_sigma];
        end
        
        %%% Fisher information matrix
        function F = fisher(obj)
            sigma = obj.theta(obj.dim+1:end);
            invsigma = diag(sigma.^-2);
            
            F_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invsigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / sigma(k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invsigma, F_blocks{:});
            
            pointSize = obj.dim;
            indices = zeros(pointSize*2,1);
            indices(1:pointSize) = 1:pointSize;
            indices(pointSize+1) = pointSize+1;
            for i = 2:pointSize
                indices(pointSize+i) = indices(pointSize+i-1) + pointSize -i + 2;
            end
            F = F(indices,indices);
        end
        
        %%% Inverse Fisher information matrix
        function invF = inverseFisher(obj)
            sigma = obj.theta(obj.dim+1:end);
            invsigma = diag(sigma.^-2);
            
            invF_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invsigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / sigma(k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(diag(sigma.^2), invF_blocks{:});
            
            pointSize = obj.dim;
            indices = zeros(pointSize*2,1);
            indices(1:pointSize) = 1:pointSize;
            indices(pointSize+1) = pointSize+1;
            for i = 2:pointSize
                indices(pointSize+i) = indices(pointSize+i-1) + pointSize -i + 2;
            end
            invF = invF(indices,indices);
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.theta(obj.dim+1:end) = 1e-8;
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            sigma = diag(obj.theta(obj.dim+1:end));
            
            params.mu = mu;
            params.Sigma = sigma.^2;
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            mu = Action * weights / sum(weights);
            sigma = zeros(obj.dim,1);
            for k = 1 : size(Action,2)
                sigma = sigma + (weights(k) * (Action(:,k) - mu).^2);
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            sigma = sigma / Z;
            sigma = sqrt(sigma);
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
