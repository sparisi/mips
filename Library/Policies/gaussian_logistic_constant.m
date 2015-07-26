%%% Gaussian with constant mean and logistic covariance: N(K,S).
%%% Params: mean and logistic weights (S = tau/(1+exp(-w)).
classdef gaussian_logistic_constant < policy
    
    properties(GetAccess = 'public', SetAccess = 'public')
        tau;
    end
    
    methods
        
        function obj = ...
                gaussian_logistic_constant(dim, init_mean, init_sigma_w, max_variance)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_sigma_w,1) == dim)
            assert(size(init_sigma_w,2) == 1)
            assert(size(max_variance,1) == size(init_sigma_w,1));
            assert(size(max_variance,2) == size(init_sigma_w,2));

            obj.theta = [init_mean; init_sigma_w];
            obj.dim   = dim;
            obj.tau   = max_variance;
            obj.dim_explore = length(init_sigma_w);
        end
        
        function probability = evaluate(obj, action)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            MU    = obj.theta(1:obj.dim);
            probability = mvnpdf(action, MU, SIGMA);
        end
        
        function action = drawAction(obj)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            MU    = obj.theta(1:obj.dim);

            action = mvnrnd(MU, SIGMA)';
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            
            % Compute covariance matrix
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            MU    = obj.theta(1:obj.dim);
            
            dlpdt = zeros(size(obj.theta));
            
            dmu = (action - MU) ./ logv;
            dlpdt(1:end-obj.dim) = dmu(:);
            
            for i = 1 : obj.dim
                wi = obj.theta(end-obj.dim+i);
                A = -0.5 * exp(-wi) / (1 + exp(-wi));
                B = 0.5 * exp(-wi) / obj.tau(i) * (action(i) - MU(i))^2;
                dlpdt(end-obj.dim+i) = A + B;
            end
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj)
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            SIGMA = diag(logv);
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(SIGMA) );
        end
        
        function obj = makeDeterministic(obj)
            obj.tau = 1e-8 * ones(size(obj.tau));
        end
        
        function params = getParams(obj)
            mu = vec2mat(obj.theta(1:obj.dim),obj.dim);
            logv  = logistic(obj.theta(end-obj.dim+1:end), obj.tau);
            sigma = diag(logv);
            
            params.mu = mu;
            params.Sigma = sigma;
        end
        
        function obj = weightedMLUpdate(obj, weights, Action)
            weights = weights / sum(weights);
            mu = Action * weights;
            logv = zeros(obj.dim,1);
            
            for j = 1 : obj.dim
                tmp = 0;
                for k = 1 : size(Action,2)
                    tmp = tmp + (weights(k) * (Action(j,k) - mu(j)).^2);
                end
                
                logv(j) = -log( obj.tau(j) / tmp - 1 );
            end
                
            obj.theta = [mu; logv];
        end
        
        function obj = randomize(obj, factor)
            obj.theta(end-obj.dim+1:end) = obj.theta(end-obj.dim+1:end) .* factor;
        end
        
        function areEq = eq(obj1, obj2)
            areEq = eq@policy(obj1,obj2);
            if max(areEq)
                areEqTau = bsxfun( @and, [obj1(:).tau], [obj2(:).tau] );
                if size(areEq,1) ~= size(areEqTau,1)
                    areEqTau = areEqTau';
                end
                areEq = bitand( areEq, areEqTau);
            else
                return;
            end
        end
        
        function plot(obj)
            params = obj.getParams;
            mu = params.mu;
            Sigma = params.Sigma;
            figure; hold all
            xlabel 'x_i'
            ylabel 'Policy density'
            x = max(abs(mu)) + 2*max(abs(Sigma(:)));
            range = -x: 0.1 : x;
            for i = 1 : length(mu)
                norm = normpdf(range, mu(i), Sigma(i,i));
                plot(range, norm)
            end            
        end
        
    end
    
end
