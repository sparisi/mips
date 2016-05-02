classdef GaussianConstantDiag < GaussianConstant
% GAUSSIANCONSTANTDIAG Gaussian distribution with constant mean and 
% diagonal covariance: N(mu,S).
% Parameters: mean mu and diagonal std s, where S = diag(s)^2.
    
    methods
        
        %% Constructor
        function obj = GaussianConstantDiag(dim, initMean, initSigma)
            assert(isscalar(dim) && ...
                size(initMean,1) == dim && ...
                size(initMean,2) == 1 && ...
                size(initSigma,1) == dim && ...
            	size(initSigma,2) == dim, ...
                'Dimensions are not consistent.')
            [~, p] = chol(initSigma);
            assert(p == 0, 'Covariance must be positive definite.')
            
            initStd = sqrt(diag(initSigma));
            obj.daction = dim;
            obj.theta = [initMean; initStd];
            obj.dparams = length(obj.theta);
            obj = obj.update(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            mu = obj.mu;
            std = sqrt(diag(obj.Sigma));
            diff = bsxfun(@minus, action, mu);
            dlogpdt_mu = bsxfun(@times, std.^-2, diff );
            dlogpdt_std = bsxfun( @plus, -std.^-1, ...
                bsxfun(@times, diff.^2, 1./std.^3) );
            dlogpdt = [dlogpdt_mu; dlogpdt_std];
        end
        
        %% FIM
        function F = fisher(obj)
        % Closed form Fisher information matrix
            std = sqrt(diag(obj.Sigma));
            invSigma = diag(std.^-2);
            
            F_blocks = cell(obj.daction,1);
            for k = 1 : obj.daction
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / std(k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invSigma, F_blocks{:});
            
            pointSize = obj.daction;
            indices = zeros(pointSize*2,1);
            indices(1:pointSize) = 1:pointSize;
            indices(pointSize+1) = pointSize+1;
            for i = 2 : pointSize
                indices(pointSize+i) = indices(pointSize+i-1) + pointSize - i + 2;
            end
            F = F(indices,indices);
        end
        
        function invF = inverseFisher(obj)
        % Closed form inverse Fisher information matrix
            std = sqrt(diag(obj.Sigma));
            invSigma = diag(std.^-2);
            
            invF_blocks = cell(obj.daction,1);
            for k = 1 : obj.daction
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / std(k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(diag(std.^2), invF_blocks{:});
            
            pointSize = obj.daction;
            indices = zeros(pointSize*2,1);
            indices(1:pointSize) = 1:pointSize;
            indices(pointSize+1) = pointSize+1;
            for i = 2 : pointSize
                indices(pointSize+i) = indices(pointSize+i-1) + pointSize -i + 2;
            end
            invF = invF(indices,indices);
        end

        %% WML
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights) >= 0, 'Weights cannot be negative.')
            weights = weights / sum(weights);
            mu = Action * weights' / sum(weights);
            std = sum(bsxfun(@times,bsxfun(@minus,Action,mu).^2,weights),2);
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            std = std / Z;
            std = sqrt(std);
            obj = obj.update([mu; std(:)]);
        end

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            mu = vec2mat(obj.theta(1:obj.daction),obj.daction);
            std = obj.theta(obj.daction+1:end);
            obj.mu = mu;
            obj.Sigma = diag(std.^2);
            obj.U = diag(std);
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.theta(end-obj.daction+1:end) = 0;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj, factor)
            obj.theta(end-obj.daction+1:end) = ... 
                obj.theta(end-obj.daction+1:end) * factor;
            obj = obj.update(obj.theta);
        end
        
    end
    
end
