classdef gaussian_chol_linear < policy_gaussian
% GAUSSIAN_CHOL_LINEAR Gaussian distribution with linear mean and constant 
% covariance: N(A*phi,S).
% Parameters: mean A and Cholesky decomposition U, with S = U'U.
%
% U is stored row-wise, e.g:
% U = [u1 u2 u3; 0 u4 u5; 0 0 u6] -> (u1 u2 u3 u4 u5 u6)
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    methods
        
        function obj = gaussian_chol_linear(basis, dim, initMean, initSigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(initMean,2))
            assert(dim == size(initMean,1))
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [initCholU, p] = chol(initSigma);
            assert(p == 0)
            
            obj.basis = basis;
            obj.dim = dim;
            init_tri = initCholU';
            init_tri = init_tri(tril(true(dim), 0)).';
            obj.theta = [initMean(:); init_tri'];
            obj.dim_explore = length(init_tri);
        end
        
        function params = getParams(obj, state)
            n = obj.dim*feval(obj.basis);
            A = vec2mat(obj.theta(1:n),obj.dim);
            indices = tril(ones(obj.dim));
            cholU = indices;
            cholU(indices == 1) = obj.theta(n+1:end);
            cholU = cholU';
            params.A = A;
            params.a = 0;
            params.cholU = cholU;
            params.Sigma = cholU'*cholU';
            if nargin == 2
                phi = feval(obj.basis, state);
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
            A = params.A;
            mu = A*phi;
            cholU = params.cholU;
            cholU = cholU';
            invU = inv(cholU);
            invSigma = invU * invU';
            
            dlogpdt_A = invSigma * (action - mu) * phi';
            
            R = invU' * (action - mu) * (action - mu)' * invSigma;
            dlogpdt_cholU = zeros(obj.dim);
            for j = 1 : obj.dim
                for i = 1 : j
                    if i == j
                        dlogpdt_cholU(i,j) = R(i,j) - 1 / cholU(i,i);
                    else
                        dlogpdt_cholU(i,j) = R(i,j);
                    end
                end
            end
            
            dlogpdt_cholU = dlogpdt_cholU';
            dlogpdt_cholU = dlogpdt_cholU(tril(true(obj.dim), 0)).';
            
            dlogpdt = [dlogpdt_A(:); dlogpdt_cholU'];
        end
        
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0) % weights cannot be negative
            Sigma = zeros(obj.dim);
            D = diag(weights);
            N = size(Action,1);
            W = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Action;
            W = W';
            for k = 1 : N
                Sigma = Sigma + (weights(k) * (Action(k,:)' - W*Phi(k,:)') * (Action(k,:)' - W*Phi(k,:)')');
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            cholU = chol(Sigma);
            tri = cholU';
            tri = tri(tril(true(obj.dim), 0)).';
            obj.theta = [W(:); tri'];
        end
        
    end
    
end
