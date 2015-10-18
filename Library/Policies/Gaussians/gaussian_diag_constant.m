classdef gaussian_diag_constant < policy_gaussian
% GAUSSIAN_DIAG_CONSTANT Gaussian distribution with constant mean and 
% diagonal covariance: N(mu,S).
% Parameters: mean mu and diagonal std s, where S = diag(s)^2.
    
    methods
        
        function obj = gaussian_diag_constant(dim, initMean, initSigma)
            assert(isscalar(dim))
            assert(size(initMean,1) == dim)
            assert(size(initMean,2) == 1)
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [~, p] = chol(initSigma);
            assert(p == 0)
            
            initStd = sqrt(diag(initSigma));
            obj.theta = [initMean; initStd];
            obj.dim = dim;
            obj.dim_explore = length(initStd);
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            std = obj.theta(obj.dim+1:end);
            params.mu = mu;
            params.Sigma = diag(std.^2);
            params.std = std;
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                dlogpdt = length(obj.theta);
                return
            end
            
            params = obj.getParams;
            mu = params.mu;
            std = params.std;
            
            dlogpdt_mu = std.^-2 .* (action - mu);
            dlogpdt_std = -std.^-1 + (action - mu).^2 ./ std.^3;
            dlogpdt = [dlogpdt_mu; dlogpdt_std];
        end
        
        %%% Fisher information matrix
        function F = fisher(obj)
            std = obj.theta(obj.dim+1:end);
            invSigma = diag(std.^-2);
            
            F_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / std(k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invSigma, F_blocks{:});
            
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
            std = obj.theta(obj.dim+1:end);
            invSigma = diag(std.^-2);
            
            invF_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / std(k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(diag(std.^2), invF_blocks{:});
            
            pointSize = obj.dim;
            indices = zeros(pointSize*2,1);
            indices(1:pointSize) = 1:pointSize;
            indices(pointSize+1) = pointSize+1;
            for i = 2:pointSize
                indices(pointSize+i) = indices(pointSize+i-1) + pointSize -i + 2;
            end
            invF = invF(indices,indices);
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights)>=0) % weights cannot be negative
            mu = Action * weights / sum(weights);
            std = zeros(obj.dim,1);
            for k = 1 : size(Action,2)
                std = std + (weights(k) * (Action(:,k) - mu).^2);
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            std = std / Z;
            std = sqrt(std);
            obj.theta = [mu; std(:)];
        end
        
    end
    
end
