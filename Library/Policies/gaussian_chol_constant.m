%%% Gaussian with constant mean and covariance: N(K,S).
%%% Params: mean and Cholesky decomposition (S = A'A).
%%% Reference: Y Sun, Efficient Natural Evolution Strategies, 2009
classdef gaussian_chol_constant < policy
    
    methods
        
        function obj = gaussian_chol_constant(dim, init_mean, init_cholA)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_cholA,1) == dim)
            assert(size(init_cholA,2) == dim)
            assert(istriu(init_cholA))
            
            obj.dim = dim;
            init_tri = init_cholA';
            init_tri = init_tri(tril(true(dim), 0)).';
            obj.theta = [init_mean; init_tri'];
            obj.dim_explore = length(init_tri);
        end
        
        function probability = evaluate(obj, action)
            mu = obj.theta(1:obj.dim);
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            probability = mvnpdf(action, mu, cholA'*cholA);
        end
        
        function action = drawAction(obj)
            mu = obj.theta(1:obj.dim);
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            action = mvnrnd(mu,cholA'*cholA)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj)
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(cholA'*cholA) );
        end
        
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            mu = obj.theta(1:obj.dim);
            
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            invsigma = cholA \ eye(size(cholA)) / cholA';
            dlogpdt_k = invsigma * (action - mu);
            
            R = cholA' \ (action - mu) * (action - mu)' * invsigma;
            dlogpdt_sigma = zeros(obj.dim);
            for i = 1 : obj.dim
                for j = i : obj.dim
                    if i == j
                        dlogpdt_sigma(i,j) = R(i,j) - 1 / cholA(i,i);
                    else
                        dlogpdt_sigma(i,j) = R(i,j);
                    end
                end
            end
            
            dlogpdt_sigma_v = dlogpdt_sigma';
            dlogpdt_sigma_v = dlogpdt_sigma_v(tril(true(obj.dim), 0)).';
            dlogpdt = [dlogpdt_k(:); dlogpdt_sigma_v'];
        end
        
        %%% Fisher information matrix
        function F = fisher(obj)
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            invsigma = cholA \ eye(size(cholA)) / cholA';
            F_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invsigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholA(k,k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invsigma, F_blocks{:});
        end
        
        %%% Inverse Fisher information matrix
        function invF = inverseFisher(obj)
            indices = tril(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            cholA = cholA';
            sigma = cholA' * cholA;
            invsigma = cholA \ eye(size(cholA)) / cholA';
            invF_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invsigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholA(k,k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(sigma, invF_blocks{:});
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.theta(obj.dim+1:end) = 1e-8;
        end
        
        function params = getParams(obj)
            mu = obj.theta(1:obj.dim);
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            sigma = cholA'*cholA;

            params.mu = mu;
            params.Sigma = sigma;
        end
        
        function obj = randomize(obj, factor)
            obj.theta(obj.dim+1:end) = obj.theta(obj.dim+1:end) .* factor;
        end
        
    end
    
end
