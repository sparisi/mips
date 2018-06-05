classdef GaussianLinearFixedvarDiagmean < GaussianLinear
% GAUSSIANLINEARFIXEDVARDIAGMEAN Gaussian distribution with linear diagonal
% mean and fixed covariance: N(A*phi,S), A diagonal matrix.
% Parameters: mean A.
    
    methods
        
        function obj = GaussianLinearFixedvarDiagmean(basis, dim, initA, Sigma)
            assert(isscalar(dim) && ...
                basis() == dim && ...
                size(initA,2) == dim && ...
                size(initA,1) == dim && ...
                size(Sigma,1) == dim && ...
                size(Sigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initU, p] = chol(Sigma);
            assert(p == 0, 'Covariance must be positive definite.')

            obj.daction = dim;
            obj.theta = diag(initA);
            obj.basis = basis;
            obj.dparams = length(obj.theta);
            obj.Sigma = Sigma;
            obj.U = initU;
            obj.A = initA;
        end
        
        %% OVERRIDE! DO NOT add the constant feature 1 (bias) to the basis function
        function phi_bias = basis_bias(obj, States)
            phi_bias = obj.basis(States);
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            phi = obj.basis(state);
            dlpdt = bsxfun(@times, (obj.Sigma) \ (action - obj.A*phi), phi);
        end
        
        %% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            phi = obj.basis(state);
            invSigma = inv(obj.Sigma);
            hlpdt = bsxfun(@times, -bsxfun(@times,permute(phi,[3 1 2]),permute(phi,[1 3 2])), invSigma);
        end
        
        %% WML
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(size(Phi,1) == obj.basis())
            assert(min(weights)>=0, 'Weights cannot be negative.')
            weights = weights / sum(weights);
            PhiW = bsxfun(@times,Phi,weights);
            tmp = PhiW * Phi';
            if rank(tmp) == size(Phi,1)
                A = tmp \ PhiW * Action';
            else
                A = pinv(tmp) * PhiW * Action';
            end
            obj = obj.update(diag(A));
        end        

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            obj.A = diag(obj.theta);
        end

        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.Sigma = 0 * obj.Sigma;
            obj.U = 0 * obj.Sigma;
        end
        
        function obj = randomize(obj,varargin)
            warning('This policy cannot be randomized!')
        end
        
    end
        
end
