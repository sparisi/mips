classdef gaussian_diag_linear < policy_gaussian
% GAUSSIAN_DIAG_LINEAR Gaussian distribution with linear mean and constant 
% diagonal covariance: N(A*phi,S).
% Parameters: mean A and diagonal std s, where S = diag(s)^2.
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    methods
        
        function obj = gaussian_diag_linear(basis, dim, initMean, initSigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(initMean,2))
            assert(dim == size(initMean,1))
            assert(size(initSigma,1) == dim)
            assert(size(initSigma,2) == dim)
            [~, p] = chol(initSigma);
            assert(p == 0)

            initStd = diag(sqrt(initSigma));
            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [initMean(:); initStd];
            obj.dim_explore = length(initStd);
        end
        
        function params = getParams(obj, state)
            n = obj.dim*feval(obj.basis);
            A = vec2mat(obj.theta(1:n),obj.dim);
            std = obj.theta(n+1:end);
            params.A = A;
            params.a = 0;
            params.Sigma = diag(std.^2);
            params.std = std;
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
            std = params.std;

            dlogpdt_A = std.^-2 .* (action - mu) * phi';
            dlogpdt_std = -std.^-1 + (action - mu).^2 ./ std.^3;

            dlogpdt = [dlogpdt_A(:); dlogpdt_std];
        end
        
        %%% Hessian of the logarithm of the policy
        function hlogpdt = hlogPidtheta(obj, state, action)
            if (nargin == 1)
                hlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis, state);
            params = obj.getParams;
            
            A = params.A;
            mu = A*phi;
            std = params.std;
            invSigma = diag(std.^-2);
            phimat = kron(phi',eye(obj.dim));
            diff = action - mu;
            
            dm = feval(obj.basis)*obj.dim;
            ds = obj.dim_explore;
            hlogpdt = zeros(length(obj.theta));
            
            % dlogpdt / (dmu dmu)
            hlogpdt(1:dm,1:dm) = - phimat' * invSigma * phimat;

            % dlogpdt / (dsigma dsigma)
            hlogpdt(dm+1:dm+ds,dm+1:dm+ds) = invSigma - 3.0 * diff.^2 * (std.^-4)';
            
            % dlogpdt / (dmu dsigma)
            hlogpdt(dm+1:dm+ds, 1:dm) = - 2 * diff * (std.^-3)' * phimat;
            hlogpdt(1:dm, dm+1:dm+ds) = hlogpdt(dm+1:dm+ds, 1:dm)';
        end
        
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0) % weights cannot be negative
            D = diag(weights);
            A = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Action;
            A = A';
            std = zeros(obj.dim,1);
            for k = 1 : size(Action,1)
                std = std + (weights(k) * (Action(k,:)' - A*Phi(k,:)').^2);
            end
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            std = std / Z;
            std = sqrt(std);
            
            obj.theta = [A(:); std];
        end
        
    end
    
end
