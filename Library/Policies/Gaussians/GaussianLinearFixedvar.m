classdef GaussianLinearFixedvar < GaussianLinear
% GAUSSIANLINEARFIXEDVAR Gaussian distribution with linear mean and fixed 
% covariance: N(A*phi,S).
% Parameters: mean A.
    
    methods
        
        %% Constructor
        function obj = GaussianLinearFixedvar(basis, dim, initA, Sigma, no_bias)
            if nargin == 4, no_bias = false; end
            obj.no_bias = no_bias; 
            assert(isscalar(dim) && ...
            	size(initA,2) == basis()+1*~no_bias && ...
            	size(initA,1) == dim && ...
                size(Sigma,1) == dim && ...
            	size(Sigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initU, p] = chol(Sigma);
            assert(p == 0, 'Covariance must be positive definite.')

            obj.daction = dim;
            obj.theta = initA(:);
            obj.basis = basis;
            obj.Sigma = Sigma;
            obj.dparams = length(obj.theta);
            obj.A = initA;
            obj.U = initU;
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            phi = obj.get_basis(state);
            dlpdt = mtimescolumn((obj.Sigma) \ (action - obj.A*phi), phi);
        end
        
        %% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            nsamples = size(state,2);
            phi = obj.get_basis(state);
            phimat = kron(phi',eye(obj.daction));
            invSigma = inv(obj.Sigma);
            hlpdt = zeros(obj.dparams,obj.dparams,nsamples);
            for i = 1 : nsamples
                idx1 = obj.daction*(i-1)+1;
                idx2 = idx1+obj.daction-1;
                subphimat = phimat(idx1:idx2,:);
                hlpdt(:,:,i) = - subphimat' * invSigma * subphimat;
            end
        end

        %% WML
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0, 'Weights cannot be negative.')
            assert(size(Phi,1) == obj.basis()+1)
            weights = weights / sum(weights);
            PhiW = bsxfun(@times,Phi,weights);
            tmp = PhiW * Phi';
            if rank(tmp) == size(Phi,1)
                A = tmp \ PhiW * Action';
            else
                A = pinv(tmp) * PhiW * Action';
            end
            A = A';
            obj = obj.update(A(:));
        end

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            A = vec2mat(obj.theta,obj.daction);
            obj.A = A;
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
