classdef GaussianLinearLogistic < GaussianLinear
% GAUSSIANLINEARLOGISTIC Gaussian distribution with linear mean and 
% logistic covariance: N(A*phi,S).
% Parameters: mean A and logistic weights w, with S = tau/(1+exp(-w)).
    
    properties
        tau
    end
    
    methods

        %% Constructor
        function obj = ...
                GaussianLinearLogistic(basis, dim, initA, initW, maxVar)
            assert(isscalar(dim) && ...
                size(initA,2) == basis()+1 && ...
                size(initA,1) == dim && ...
                size(initW,1) == dim && ...
                size(initW,2) == 1 && ...
                size(maxVar,1) == size(initW,1) && ...
                size(maxVar,2) == size(initW,2), ...
            'Dimensions are not consistent.')

            obj.daction = dim;
            obj.basis = basis;
            obj.tau = maxVar;
            obj.theta = [initA(:); initW];
            obj.dparams = length(obj.theta);
            obj = obj.update(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            nsamples = size(state,2);
            logv = logistic(obj.theta(end-obj.daction+1:end), obj.tau);
            phi = obj.basis_bias(state);
            mu = obj.A * phi;
            dlpdt = zeros(obj.dparams,nsamples);
            diff = action - mu;

            % derivative wrt the mean
            dlpdt(1:end-obj.daction,:) = mtimescolumn( ...
                bsxfun(@times, 1./logv, diff), phi );

            % derivative wrt the logistic covariance
            w = obj.theta(end-obj.daction+1:end);
            ds1 = -0.5 .* exp(-w) ./ (1 + exp(-w));
            ds2 = bsxfun(@times, 0.5 .* exp(-w) ./ obj.tau, diff.^2);
            dlpdt(end-obj.daction+1:end,:) = bsxfun(@plus, ds1, ds2);
        end

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            logv = logistic(obj.theta(end-obj.daction+1:end), obj.tau);
            Sigma = diag(logv);
            A = vec2mat(obj.theta(1:end-obj.daction),obj.daction);
            obj.A = A;
            obj.Sigma = Sigma;
            obj.U = diag(sqrt(logv));
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.theta(end-obj.daction+1:end) = -1e8;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj, varargin)
            warning('Not implemented for this policy!')
        end
        
    end
    
end
