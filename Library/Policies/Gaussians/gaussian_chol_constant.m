classdef gaussian_chol_constant < policy_gaussian
% GAUSSIAN_CHOL_CONSTANT Gaussian distribution with constant mean and 
% covariance: N(mu,S).
% Parameters: mean mu and Cholesky decomposition U, with S = U'U.
%
% U is stored row-wise, e.g:
% U = [u1 u2 u3; 0 u4 u5; 0 0 u6] -> (u1 u2 u3 u4 u5 u6)
%
% =========================================================================
% REFERENCE
% Y Sun, D Wierstra, T Schaul, J Schmidhuber
% Efficient Natural Evolution Strategies (2009)
    
    methods
        
        function obj = gaussian_chol_constant(dim, initMean, initSigma)
            assert(isscalar(dim))
            assert(size(initMean,1) == dim)
            assert(size(initMean,2) == 1)
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [initCholU, p] = chol(initSigma);
            assert(p == 0)

            obj.dim = dim;
            init_tri = initCholU';
            init_tri = init_tri(tril(true(dim), 0)).';
            obj.theta = [initMean; init_tri'];
            obj.dim_explore = length(init_tri);
        end
        
        function params = getParams(obj)
            mu = obj.theta(1:obj.dim);
            indices = tril(ones(obj.dim));
            cholU = indices;
            cholU(indices == 1) = obj.theta(obj.dim+1:end);
            cholU = cholU';
            params.mu = mu;
            params.Sigma = cholU'*cholU;
            params.cholU = cholU;
        end
        
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            params = obj.getParams;
            mu = params.mu;
            cholU = params.cholU;
            invU = inv(cholU);
            invSigma = invU * invU';

            dlogpdt_A = invSigma * (action - mu);
            
            R = invU' * (action - mu) * (action - mu)' * invSigma;
            dlogpdt_cholU = zeros(obj.dim);
            for i = 1 : obj.dim
                for j = i : obj.dim
                    if i == j
                        dlogpdt_cholU(i,j) = R(i,j) - 1 / cholU(i,i);
                    else
                        dlogpdt_cholU(i,j) = R(i,j);
                    end
                end
            end
            
            dlogpdt_cholUT = dlogpdt_cholU';
            dlogpdt_cholUT = dlogpdt_cholUT(tril(true(obj.dim), 0)).';
            dlogpdt = [dlogpdt_A(:); dlogpdt_cholUT'];
        end
        
        %%% Fisher information matrix
        function F = fisher(obj)
            params = obj.getParams;
            cholU = params.cholU;
            invU = inv(cholU);
            invSigma = invU * invU';
            F_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholU(k,k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invSigma, F_blocks{:});
        end
        
        %%% Inverse Fisher information matrix
        function invF = inverseFisher(obj)
            params = obj.getParams;
            cholU = params.cholU;
            Sigma = cholU' * cholU;
            invU = inv(cholU);
            invSigma = invU * invU';
            invF_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholU(k,k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(Sigma, invF_blocks{:});
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights)>=0) % weights cannot be negative
            mu = Action * weights / sum(weights);
            Sigma = zeros(size(obj.dim));
            for k = 1 : size(Action,2)
                Sigma = Sigma + (weights(k) * (Action(:,k) - mu) * (Action(:,k) - mu)');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            cholU = chol(Sigma);
            tri = cholU';
            tri = tri(tril(true(obj.dim), 0)).';
            obj.theta = [mu; tri'];
        end
        
    end
    
end
