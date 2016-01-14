classdef GaussianLinearDiag < GaussianLinear
% GAUSSIANLINEARDIAG Gaussian distribution with linear mean and constant 
% diagonal covariance: N(A*phi,S).
% Parameters: mean A and diagonal std s, where S = diag(s)^2.
    
    methods

        %% Constructor
        function obj = GaussianLinearDiag(basis, dim, initA, initSigma)
            assert(isscalar(dim) && ...
            	size(initA,2) == basis()+1 && ...
            	size(initA,1) == dim && ...
                size(initSigma,1) == dim && ...
            	size(initSigma,2) == dim, ...
                'Dimensions are not consistent.')
            [~, p] = chol(initSigma);
            assert(p == 0, 'Covariance must be positive definite.')

            obj.daction = dim;
            initStd = diag(sqrt(initSigma));
            obj.basis = basis;
            obj.theta = [initA(:); initStd];
            obj.dparams = length(obj.theta);
            obj = obj.update([initA(:); initStd]);
        end
        
        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            phi = obj.basis1(state);
            A = obj.A;
            mu = A*phi;
            std = sqrt(diag(obj.Sigma));
            
            dlogpdt_A = mtimescolumn( ...
                bsxfun(@times, std.^-2, action - mu), phi );
            dlogpdt_std = bsxfun( @plus, -std.^-1, ...
                bsxfun(@times, (action - mu).^2, 1./std.^3) );
            dlogpdt = [dlogpdt_A; dlogpdt_std];
        end
        
        %% Hessian of the logarithm of the policy
        function hlogpdt = hlogPidtheta(obj, state, action)
            nsamples = size(state,2);
            phi = obj.basis1(state);
            dphi = size(phi,1);
            A = obj.A;
            mu = A*phi;
            std = sqrt(diag(obj.Sigma));
            invSigma = diag(std.^-2);
            phimat = kron(phi',eye(obj.daction));
            diff = action - mu;
            
            dm = dphi*obj.daction;
            ds = obj.dparams - numel(obj.A);
            hlogpdt = zeros(obj.dparams,obj.dparams,nsamples);

            for i = 1 : nsamples
                idx1 = dphi*(i-1)+1;
                idx2 = idx1+dphi-1;
                subphimat = phimat(idx1:idx2,:);

                % dlogpdt / (dmu dmu)
                hlogpdt(1:dm,1:dm,i) = - subphimat' * invSigma * subphimat;

                % dlogpdt / (dsigma dsigma)
                hlogpdt(dm+1:dm+ds,dm+1:dm+ds,i) = invSigma - 3.0 * diff(:,i).^2 * (std.^-4)';

                % dlogpdt / (dmu dsigma)
                hlogpdt(dm+1:dm+ds, 1:dm, i) = - 2 * diff(:,i) * (std.^-3)' * subphimat;
                hlogpdt(1:dm, dm+1:dm+ds, i) = hlogpdt(dm+1:dm+ds, 1:dm, i)';
            end
        end
        
        %% WML
        function obj = weightedMLUpdate(obj, weights, Action, Phi)
            assert(min(weights)>=0, 'Weights cannot be negative.')
            D = diag(weights);
            A = (Phi * D * Phi' + 1e-8 * eye(size(Phi,1))) \ Phi * D * Action';
            A = A';
            std = sum( bsxfun(@times, weights, bsxfun(@minus, Action, A*Phi).^2), 2 );
            Z = (sum(weights)^2 - sum(weights.^2)) / sum(weights);
            std = std / Z;
            std = sqrt(std);
            
            obj = obj.update([A(:); std]);
        end
        
        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            n = length(obj.theta) - obj.daction;
            A = vec2mat(obj.theta(1:n),obj.daction);
            std = obj.theta(n+1:end);
            obj.A = A;
            obj.Sigma = diag(std.^2);
            obj.U = diag(std);
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.theta(end-obj.daction+1:end) = 1e-4;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj)
            obj.theta(end-obj.daction+1:end) = ... 
                obj.theta(end-obj.daction+1:end) * factor;
            obj = obj.update(obj.theta);
        end
        
    end
    
end
