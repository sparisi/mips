classdef GaussianConstantChol < GaussianConstant
% GAUSSIANCONSTANTCHOL Gaussian distribution with constant mean and 
% covariance: N(mu,S).
% Parameters: mean mu and Cholesky decomposition U, with S = U'U.
%
% U is stored row-wise, e.g:
% U = [u1 u2 u3; 
%      0  u4 u5; 
%      0  0  u6] -> (u1 u2 u3 u4 u5 u6)
%
% =========================================================================
% REFERENCE
% Y Sun, D Wierstra, T Schaul, J Schmidhuber
% Efficient Natural Evolution Strategies (2009)
    
    methods

        %% Constructor
        function obj = GaussianConstantChol(dim, initMean, initSigma)
            assert(isscalar(dim) && ...
                size(initMean,1) == dim && ...
                size(initMean,2) == 1 && ...
                size(initSigma,1) == dim && ...
            	size(initSigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initCholU, p] = chol(initSigma);
            assert(p == 0, 'Covariance must be positive definite.')

            obj.daction = dim;
            init_tri = initCholU';
            init_tri = init_tri(tril(true(dim), 0)).';
            obj.theta = [initMean; init_tri'];
            obj.dparams = length(obj.theta);
            obj = obj.update(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            nsamples = size(action,2);
            mu = obj.mu;
            cholU = obj.U;
            invU = inv(cholU);
            invUT = invU';
            invSigma = invU * invU';
            diff = bsxfun(@minus,action,mu);

            dlogpdt_A = invSigma * diff;

            tmp = bsxfun(@plus,-invUT(:),mtimescolumn(invUT*diff, invSigma*diff));
            tmp = reshape(tmp,obj.daction,obj.daction,nsamples);
            tmp = permute(tmp,[2 1 3]);
            idx = tril(true(obj.daction));
            nelements = sum(sum(idx));
            idx = repmat(idx,1,1,nsamples);
            dlogpdt_cholU = tmp(idx);
            dlogpdt_cholU = reshape(dlogpdt_cholU,nelements,nsamples);
            
            dlogpdt = [dlogpdt_A; dlogpdt_cholU];
        end

        %% WML
        function obj = weightedMLUpdate(obj, weights, Action)
            assert(min(weights) >= 0, 'Weights cannot be negative.')
            mu = Action * weights' / sum(weights);
            diff = bsxfun(@minus,Action,mu);
            Sigma = bsxfun(@times, diff, weights) * diff';
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            cholU = chol(Sigma);
            tri = cholU';
            tri = tri(tril(true(obj.daction), 0)).';
            obj = obj.update([mu; tri']);
        end

        %% FIM
        function F = fisher(obj)
        % Closed form Fisher information matrix
            cholU = obj.U;
            invU = inv(cholU);
            invSigma = invU * invU';

            F = zeros(sum(1:obj.daction)+obj.daction);
            F(1:obj.daction,1:obj.daction) = invSigma;
            idx = obj.daction+1;
            for k = 1 : obj.daction
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholU(k,k)^2;
                step = size(tmp,1);
                F(idx:idx+step-1,idx:idx+step-1) = tmp;
                idx = idx + step;
            end
        end
        
        function invF = inverseFisher(obj)
        % Closed form inverse Fisher information matrix
            cholU = obj.U;
            invU = inv(cholU);
            invSigma = invU * invU';
            
            invF = zeros(sum(1:obj.daction)+obj.daction);
            invF(1:obj.daction,1:obj.daction) = obj.Sigma;
            idx = obj.daction+1;
            for k = 1 : obj.daction
                tmp = invSigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholU(k,k)^2;
                step = size(tmp,1);
                invF(idx:idx+step-1,idx:idx+step-1) = eye(size(tmp)) / tmp;
                idx = idx + step;
            end
        end
        
        %% Update
        function obj = update(obj, varargin)
            if nargin == 2 % Update by params
                theta = varargin{1};
                obj.theta(1:length(theta)) = theta;
                mu = obj.theta(1:obj.daction);
                indices = tril(ones(obj.daction));
                cholU = indices;
                cholU(indices == 1) = obj.theta(obj.daction+1:end);
                cholU = cholU';
                obj.mu = mu;
                obj.Sigma = cholU'*cholU;
                obj.U = cholU;
            elseif nargin == 3 % Update by mean and covariance
                obj.mu = varargin{1};
                obj.Sigma = varargin{2};
                [U, p] = chol(varargin{2});
                assert(p == 0, 'Covariance must be positive definite.')
                U = U';
                U = U(tril(true(obj.daction), 0)).';
                obj.theta = [obj.mu; U'];
            else
                error('Wrong number of input arguments')
            end
        end

        %% Change stochasticity
        function obj = makeDeterministic(obj)
            n = obj.daction;
            obj.theta(n+1:end) = obj.theta(n+1:end) * 1e-4;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj, factor)
            n = obj.daction;
            obj.theta(n+1:end) = obj.theta(n+1:end) * factor;
            obj = obj.update(obj.theta);
        end
        
    end
    
end
