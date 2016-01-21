classdef GaussianConstantFixedvar < GaussianConstant
% GAUSSIANCONSTANTFIXEDVAR Gaussian distribution with constant mean and 
% fixed covariance: N(mu,S).
% Parameters: mean mu.
    
    methods
        
        %% Constructor
        function obj = GaussianConstantFixedvar(dim, initMean, Sigma)
            assert(isscalar(dim) && ...
                size(initMean,1) == dim && ...
                size(initMean,2) == 1 && ...
                size(Sigma,1) == dim && ...
            	size(Sigma,2) == dim, ...
                'Dimensions are not consistent.')
            [initU, p] = chol(Sigma);
            assert(p == 0, 'Covariance must be positive definite.')
            
            obj.theta = initMean;
            obj.daction = dim;
            obj.Sigma = Sigma;
            obj.U = initU;
            obj.mu = initMean;
            obj.dparams = length(obj.theta);
        end
        
        %% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, action)
            mu = obj.mu;
            dlogpdt = obj.Sigma \ bsxfun(@minus, action, mu);
        end
        
        %% Hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            hlpdt = -inv(obj.Sigma);
        end

        %% WML
        function obj = weightedMLUpdate(obj, weights, Actions)
            assert(min(weights) >= 0, 'Weights cannot be negative.')
            mu = Actions * weights / sum(weights);
            obj = obj.update(mu);
        end        

        %% Update
        function obj = update(obj, theta)
            obj.theta(1:length(theta)) = theta;
            obj.mu = vec2mat(obj.theta(1:obj.daction),obj.daction);
        end
        
        %% Change stochasticity
        function obj = makeDeterministic(obj)
            obj.Sigma = 1e-8 * obj.Sigma;
            obj.U = chol(obj.Sigma);
        end
        
        function obj = randomize(obj)
            warning('This policy cannot be randomized!')
        end
        
    end
    
end
