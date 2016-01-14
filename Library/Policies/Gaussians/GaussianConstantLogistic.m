classdef GaussianConstantLogistic < GaussianConstant
% GAUSSIANCONSTANTLOGISTIC Gaussian distribution with constant mean and 
% logistic covariance: N(mu,S).
% Parameters: mean mu and logistic weights w, with S = tau/(1+exp(-w)).
    
    properties
        tau
    end
    
    methods
        
        %% Constructor
        function obj = ...
                GaussianConstantLogistic(dim, initMean, initW, maxVar)
            assert(isscalar(dim) && ...
            	size(initMean,1) == dim && ...
                size(initMean,2) == 1 && ...
            	size(initW,1) == dim && ...
            	size(initW,2) == 1 && ...
            	size(maxVar,1) == size(initW,1) && ...
            	size(maxVar,2) == size(initW,2), ...
                'Dimensions are not consistent.')

            obj.daction = dim;
            obj.tau = maxVar;
            obj.theta = [initMean; initW];
            obj.dparams = length(obj.theta);
            obj = obj.update(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, action)
            logv = logistic(obj.theta(end-obj.daction+1:end), obj.tau);
            mu = obj.mu;
            dlpdt = zeros(obj.dparams, size(action,2));

            % derivative wrt the mean
            diff = bsxfun(@minus,action,mu);
            dlpdt(1:end-obj.daction,:) = bsxfun(@times, 1./logv, diff);
            
            % derivative wrt the logistic covariance
            w = obj.theta(end-obj.daction+1:end);
            ds1 = -0.5 .* exp(-w) ./ (1 + exp(-w));
            ds2 = bsxfun(@times, 0.5 .* exp(-w) ./ obj.tau, diff.^2);
            dlpdt(end-obj.daction+1:end,:) = bsxfun(@plus, ds1, ds2);
        end
        
        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            mu = vec2mat(obj.theta(1:obj.daction),obj.daction);
            logv = logistic(obj.theta(end-obj.daction+1:end), obj.tau);
            Sigma = diag(logv);
            obj.mu = mu;
            obj.Sigma = Sigma;
            obj.U = diag(sqrt(diag(Sigma)));
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.theta(end-obj.daction+1:end) = -1e8;
            obj = obj.update(obj.theta);
        end
        
        function obj = randomize(obj)
            warning('Not implemented for this policy!')
        end
        
    end
    
end
